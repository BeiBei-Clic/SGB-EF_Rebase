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
import numpy as np
from typing import List, Tuple
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


def generate_dataset(
    num_samples: int = 1000000,
    n_points: int = 500,
    x_range: Tuple[float, float] = (-10, 10),
    num_variables: int = 3,
    max_depth: int = 3,
) -> dict:
    """生成符号回归数据集.

    Returns:
        data: 包含所有NPZ键的字典
    """
    # ========== 操作 1: 初始化存储容器 ==========
    input_dimensions_list = []
    x_values_list = []
    y_target_list = []
    z0_token_ids_list = []
    z1_token_ids_list = []
    x0_token_ids_list = []
    x1_token_ids_list = []

    # ========== 操作 2: 初始化 Vocabulary 和 RNG ==========
    vocab = Vocabulary(num_variables=num_variables)
    rng = np.random.default_rng()

    # ========== 操作 3: 生成样本循环 ==========
    samples_collected = 0

    while samples_collected < num_samples:

        # 3.1 生成表达式、采样点并计算目标值
        result = generate_expression_sample(
            num_variables=num_variables,
            max_depth=max_depth,
            n_points=n_points,
            x_range=x_range,
            vocab=vocab,
            rng=rng,
        )

        if result is False:
            continue

        base_expr, target_expr, x_i, y_i = result

        # 3.2 将表达式转换为 token 序列
        base_tokens = expression_to_tokens(base_expr)
        target_tokens = expression_to_tokens(target_expr)

        # 3.3 将 token 字符串转换为 token ids
        base_token_ids = vocab.encode(base_tokens)
        target_token_ids = vocab.encode(target_tokens)

        # 3.4 创建编辑路径（生成 z0, z1, x0, x1）
        z0_ids, z1_ids, x0_ids, x1_ids = create_edit_path(
            target_token_ids,
            base_token_ids=base_token_ids,
            vocab=vocab,
        )

        # 3.5 收集当前样本数据
        input_dimensions_list.append(num_variables)
        x_values_list.append(x_i)
        y_target_list.append(y_i)
        z0_token_ids_list.append(np.array(z0_ids, dtype=np.int64))
        z1_token_ids_list.append(np.array(z1_ids, dtype=np.int64))
        x0_token_ids_list.append(np.array(x0_ids, dtype=np.int64))
        x1_token_ids_list.append(np.array(x1_ids, dtype=np.int64))

        samples_collected += 1

        # 3.6 如果删减成功（x0 < x1），生成反向样本
        if len(x0_ids) < len(x1_ids) and samples_collected < num_samples:
            # 调换 x0, x1 和 z0, z1
            x0_ids, x1_ids = x1_ids, x0_ids
            z0_ids, z1_ids = z1_ids, z0_ids

            # 使用 base_expr 重新计算 y 值
            y_base = evaluate_expression(base_expr, x_i)

            if y_base is not False:
                # 收集反向样本数据
                input_dimensions_list.append(num_variables)
                x_values_list.append(x_i)
                y_target_list.append(y_base)
                z0_token_ids_list.append(np.array(z0_ids, dtype=np.int64))
                z1_token_ids_list.append(np.array(z1_ids, dtype=np.int64))
                x0_token_ids_list.append(np.array(x0_ids, dtype=np.int64))
                x1_token_ids_list.append(np.array(x1_ids, dtype=np.int64))

                samples_collected += 1

    # ========== 操作 4: 堆叠数组并转换为 numpy 格式 ==========
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

    # ========== 操作 5: 构建返回字典 ==========
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
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（默认: data/data_{num_samples}_{num_variables}v.npz）")

    args = parser.parse_args()

    # 默认输出文件名格式: data/data_{num_samples}_{num_variables}v.npz
    if args.output is None:
        args.output = f"data/data_{args.num_samples}_{args.num_variables}v.npz"

    data = generate_dataset(
        num_samples=args.num_samples,
        n_points=args.n_points,
        x_range=(args.x_min, args.x_max),
        num_variables=args.num_variables,
        max_depth=args.max_depth,
    )

    save_dataset(args.output, data)
    print(f"已生成数据集，样本数: {len(data['input_dimensions'])}，保存至: {args.output}")
