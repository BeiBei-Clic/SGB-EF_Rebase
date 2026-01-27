"""Data generator for symbolic regression using SymPy.

数据格式与 data_loader.py 兼容：
- input_dimensions: 输入维度
- x_values: (num_samples, n_points, input_dim) 回归输入特征
- y_target: (num_samples, n_points) 回归目标值
- z0_token_ids: (num_samples, seq_len) 对齐的base序列（含gap token）
- z1_token_ids: (num_samples, seq_len) 对齐的target序列（含gap token）
- x0_token_ids: (num_samples, seq_len) 非对齐的base序列（无gap token）
- x1_token_ids: (num_samples, seq_len) 非对齐的target序列（无gap token）
"""
import argparse
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
from data_generator.expression_generator import generate_expression_sample, expression_to_tokens, evaluate_expression
from data_generator.edit_path import create_edit_path
from src.model.vocab import Vocabulary


def save_dataset(path: str, data: dict) -> None:
    """保存生成数据为NPZ格式.

    Args:
        path: 保存路径
        data: 包含以下键的字典:
            - input_dimensions: (num_samples,) 输入维度
            - x_values: (num_samples, n_points, num_variables) 输入特征
            - y_target: (num_samples, n_points) 目标值
            - z0_token_ids: (num_samples, seq_len) 对齐base序列
            - z1_token_ids: (num_samples, seq_len) 对齐target序列
            - x0_token_ids: (num_samples, seq_len) 非对齐base序列
            - x1_token_ids: (num_samples, seq_len) 非对齐target序列
    """
    np.savez(path, **data)


def generate_single_sample(
    seed: int,
    num_variables: int,
    max_depth: int,
    n_points: int,
    x_range: Tuple[float, float],
) -> Optional[dict]:
    """生成单个样本（用于并行处理）.

    Args:
        seed: 随机种子
        num_variables: 变量数量
        max_depth: 表达式最大深度
        n_points: 每个样本的采样点数量
        x_range: 采样范围

    Returns:
        样本字典（含反向样本）或 None（生成失败）
    """
    rng = np.random.default_rng(seed)
    vocab = Vocabulary(num_variables=num_variables)

    result = generate_expression_sample(
        num_variables=num_variables,
        max_depth=max_depth,
        n_points=n_points,
        x_range=x_range,
        vocab=vocab,
        rng=rng,
    )

    if result is False:
        return None

    base_expr, target_expr, x_i, y_i = result

    base_tokens = expression_to_tokens(base_expr)
    target_tokens = expression_to_tokens(target_expr)

    base_token_ids = vocab.encode(base_tokens)
    target_token_ids = vocab.encode(target_tokens)

    z0_ids, z1_ids, x0_ids, x1_ids = create_edit_path(
        target_token_ids,
        base_token_ids=base_token_ids,
        vocab=vocab,
    )

    sample = {
        "input_dimension": num_variables,
        "x_values": x_i,
        "y_target": y_i,
        "z0_token_ids": np.array(z0_ids, dtype=np.int64),
        "z1_token_ids": np.array(z1_ids, dtype=np.int64),
        "x0_token_ids": np.array(x0_ids, dtype=np.int64),
        "x1_token_ids": np.array(x1_ids, dtype=np.int64),
        "has_reverse": False,
        "reverse_sample": None,
    }

    # 如果删减成功（x0 < x1），生成反向样本
    if len(x0_ids) < len(x1_ids):
        x0_ids_rev, x1_ids_rev = x1_ids, x0_ids
        z0_ids_rev, z1_ids_rev = z1_ids, z0_ids

        y_base = evaluate_expression(base_expr, x_i)

        if y_base is not False:
            sample["has_reverse"] = True
            sample["reverse_sample"] = {
                "input_dimension": num_variables,
                "x_values": x_i,
                "y_target": y_base,
                "z0_token_ids": np.array(z0_ids_rev, dtype=np.int64),
                "z1_token_ids": np.array(z1_ids_rev, dtype=np.int64),
                "x0_token_ids": np.array(x0_ids_rev, dtype=np.int64),
                "x1_token_ids": np.array(x1_ids_rev, dtype=np.int64),
            }

    return sample

def generate_dataset(
    num_samples: int = 1000000,
    n_points: int = 500,
    x_range: Tuple[float, float] = (-10, 10),
    num_variables: int = 3,
    max_depth: int = 3,
    num_workers: int = -1,
) -> dict:
    """生成符号回归数据集（支持多核并行）.

    Args:
        num_samples: 目标样本数量
        n_points: 每个样本的采样点数量
        x_range: 采样范围
        num_variables: 变量数量
        max_depth: 表达式最大深度
        num_workers: 并行进程数（-1表示使用所有CPU核心）

    Returns:
        data: 包含所有NPZ键的字典
    """
    # 清理并重新创建日志文件
    log_path = Path("logs/data_generator.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    # 初始化存储容器
    input_dimensions_list = []
    x_values_list = []
    y_target_list = []
    z0_token_ids_list = []
    z1_token_ids_list = []
    x0_token_ids_list = []
    x1_token_ids_list = []

    rng = np.random.default_rng()
    vocab = Vocabulary(num_variables=num_variables)

    samples_collected = 0
    batch_size = max(1000, num_samples // (num_workers * 10 if num_workers > 0 else 100))

    with tqdm(total=num_samples, desc="生成样本") as pbar:
        while samples_collected < num_samples:
            pbar.n = samples_collected
            pbar.refresh()
            # 计算本次需要生成的样本数
            remaining = num_samples - samples_collected
            current_batch = min(batch_size, remaining)

            # 生成随机种子
            seeds = rng.integers(0, 2**32, size=current_batch)

            # 并行生成样本
            results = Parallel(n_jobs=num_workers)(
                delayed(generate_single_sample)(
                    seed=int(seed),
                    num_variables=num_variables,
                    max_depth=max_depth,
                    n_points=n_points,
                    x_range=x_range,
                )
                for seed in seeds
            )

            # 收集成功生成的样本
            for sample in results:
                if sample is None:
                    continue

                input_dimensions_list.append(sample["input_dimension"])
                x_values_list.append(sample["x_values"])
                y_target_list.append(sample["y_target"])
                z0_token_ids_list.append(sample["z0_token_ids"])
                z1_token_ids_list.append(sample["z1_token_ids"])
                x0_token_ids_list.append(sample["x0_token_ids"])
                x1_token_ids_list.append(sample["x1_token_ids"])

                samples_collected += 1

                # 处理反向样本
                if sample["has_reverse"] and samples_collected < num_samples:
                    rev = sample["reverse_sample"]
                    input_dimensions_list.append(rev["input_dimension"])
                    x_values_list.append(rev["x_values"])
                    y_target_list.append(rev["y_target"])
                    z0_token_ids_list.append(rev["z0_token_ids"])
                    z1_token_ids_list.append(rev["z1_token_ids"])
                    x0_token_ids_list.append(rev["x0_token_ids"])
                    x1_token_ids_list.append(rev["x1_token_ids"])

                    samples_collected += 1

                if samples_collected >= num_samples:
                    break

    # 堆叠数组并转换为 numpy 格式
    input_dimensions = np.array(input_dimensions_list, dtype=np.int32)
    x_values = np.stack(x_values_list, axis=0)
    y_target = np.stack(y_target_list, axis=0)

    # 对 token 序列进行 padding
    pad_token_id = vocab.pad_token
    max_seq_len = max(
        max(len(ids) for ids in z0_token_ids_list),
        max(len(ids) for ids in z1_token_ids_list),
        max(len(ids) for ids in x0_token_ids_list),
        max(len(ids) for ids in x1_token_ids_list),
    )

    def pad_sequence(ids_list: List[np.ndarray], max_len: int, pad_value: int) -> np.ndarray:
        padded = []
        for ids in ids_list:
            padded_ids = np.full(max_len, pad_value, dtype=np.int64)
            padded_ids[:len(ids)] = ids
            padded.append(padded_ids)
        return np.stack(padded, axis=0)

    z0_token_ids = pad_sequence(z0_token_ids_list, max_seq_len, pad_token_id)
    z1_token_ids = pad_sequence(z1_token_ids_list, max_seq_len, pad_token_id)
    x0_token_ids = pad_sequence(x0_token_ids_list, max_seq_len, pad_token_id)
    x1_token_ids = pad_sequence(x1_token_ids_list, max_seq_len, pad_token_id)

    # 构建返回字典
    data = {
        "input_dimensions": input_dimensions,
        "x_values": x_values,
        "y_target": y_target,
        "z0_token_ids": z0_token_ids,
        "z1_token_ids": z1_token_ids,
        "x0_token_ids": x0_token_ids,
        "x1_token_ids": x1_token_ids,
    }

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成符号回归数据集")
    parser.add_argument("--num-samples", type=int, default=1000, help="生成样本数量")
    parser.add_argument("--n-points", type=int, default=100, help="每个样本的采样点数量")
    parser.add_argument("--x-min", type=float, default=-10, help="采样范围最小值")
    parser.add_argument("--x-max", type=float, default=10, help="采样范围最大值")
    parser.add_argument("--num-variables", type=int, default=3, help="变量数量")
    parser.add_argument("--max-depth", type=int, default=3, help="表达式最大深度")
    parser.add_argument("--num-workers", type=int, default=-1, help="并行进程数（-1表示使用所有CPU核心，1表示单核）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（默认: data/data_{num_samples}_{num_variables}v.npz）")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/data_{args.num_samples}_{args.num_variables}v.npz"

    data = generate_dataset(
        num_samples=args.num_samples,
        n_points=args.n_points,
        x_range=(args.x_min, args.x_max),
        num_variables=args.num_variables,
        max_depth=args.max_depth,
        num_workers=args.num_workers,
    )

    save_dataset(args.output, data)
    print(f"已生成数据集，样本数: {len(data['input_dimensions'])}，保存至: {args.output}")
