"""Edit Flow 采样推理模块。

实现基于 Poisson 过程的编辑流采样，用于符号回归表达式生成。
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.data_embedding import SetEncoder, prepare_encoder_input
from src.model.vocab import Vocabulary
from src.utils.checkpoint import CheckpointManager
from src.core.flow_helper import CubicScheduler


def get_adaptive_h(default_h: float, t: torch.Tensor, scheduler: CubicScheduler) -> torch.Tensor:
    """计算自适应时间步长。

    根据当前时间步和调度器动态调整步长，避免在 kappa(t) 接近 1 时步长过大。

    Args:
        default_h: 默认步长
        t: 当前时间 (batch_size, 1)
        scheduler: Kappa 调度器

    Returns:
        adapt_h: 自适应步长 (batch_size, 1)
    """
    kappa_t = scheduler(t)
    # 只在 kappa(t) 非常接近 1 时才减小步长
    adapt_h = torch.where(
        kappa_t > 0.99,
        default_h * 0.5,
        default_h
    )
    # 同时确保最后一步不会超过 1
    adapt_h = torch.minimum(adapt_h, 1 - t)
    return adapt_h


def apply_ins_del_operations(
    x_t: torch.Tensor,
    ins_mask: torch.Tensor,
    del_mask: torch.Tensor,
    sub_mask: torch.Tensor,
    ins_probs: torch.Tensor,
    sub_probs: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, int, int, int]:
    """应用插入、删除、替换操作到序列。

    正确处理所有编辑操作类型，使用模型预测的 token。

    Args:
        x_t: 当前序列 (batch_size, seq_len)
        ins_mask: 插入掩码 (batch_size, seq_len)
        del_mask: 删除掩码 (batch_size, seq_len)
        sub_mask: 替换掩码 (batch_size, seq_len)
        ins_probs: 插入 token 概率分布 (batch_size, seq_len, vocab_size)
        sub_probs: 替换 token 概率分布 (batch_size, seq_len, vocab_size)
        pad_token_id: 填充 token ID

    Returns:
        new_x_t: 操作后的序列
        n_ins: 插入操作数量
        n_del: 删除操作数量
        n_sub: 替换操作数量
    """
    batch_size, seq_len = x_t.shape
    device = x_t.device

    new_sequences = []
    total_ins, total_del, total_sub = 0, 0, 0

    for b in range(batch_size):
        seq = x_t[b].clone()

        # 找到非 pad token 的位置
        non_pad_indices = (seq != pad_token_id).nonzero(as_tuple=True)[0]
        actual_len = len(non_pad_indices) if len(non_pad_indices) > 0 else 0

        if actual_len == 0:
            new_sequences.append(seq)
            continue

        # 处理删除操作（优先处理）
        batch_del_mask = del_mask[b, :actual_len]
        del_positions = batch_del_mask.nonzero(as_tuple=True)[0]
        for pos in del_positions:
            seq[pos] = pad_token_id
            total_del += 1

        # 处理替换操作
        batch_sub_mask = sub_mask[b, :actual_len]
        sub_positions = batch_sub_mask.nonzero(as_tuple=True)[0]
        for i, pos in enumerate(sub_positions):
            # 从概率分布采样 token
            probs = sub_probs[b, pos]
            token = torch.multinomial(probs, 1).item()
            seq[pos] = token
            total_sub += 1

        # 处理插入操作
        batch_ins_mask = ins_mask[b, :actual_len]
        ins_positions = batch_ins_mask.nonzero(as_tuple=True)[0]
        for i, pos in enumerate(ins_positions):
            if actual_len < seq_len:
                # 从概率分布采样 token
                probs = ins_probs[b, pos]
                token = torch.multinomial(probs, 1).item()
                seq[actual_len] = token
                actual_len += 1
                total_ins += 1

        new_sequences.append(seq)

    return torch.stack(new_sequences, dim=0), total_ins, total_del, total_sub


def load_test_sample(npz_path: str, sample_idx: int, device: torch.device) -> Tuple:
    """从 NPZ 文件加载指定样本。

    Args:
        npz_path: NPZ 文件路径
        sample_idx: 样本索引
        device: 设备

    Returns:
        x_values: 输入特征 (n_points, 3)
        y_target: 目标值 (n_points,)
        x0_token_ids: 初始 token 序列
    """
    data = np.load(npz_path, allow_pickle=True)

    x_values = torch.from_numpy(data['x_values'][sample_idx]).float().to(device)
    y_target = torch.from_numpy(data['y_target'][sample_idx]).float().to(device)
    x0_token_ids = torch.from_numpy(data['x0_token_ids'][sample_idx]).long().to(device)

    return x_values, y_target, x0_token_ids


def edit_flow_sampling(
    model: EditFlowsTransformer,
    data_encoder: SetEncoder,
    x_values: torch.Tensor,
    y_target: torch.Tensor,
    x0_token_ids: torch.Tensor,
    vocab: Vocabulary,
    scheduler: CubicScheduler,
    n_steps: int = 1000,
    default_h: float = 0.1,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """执行编辑流采样（Euler 方法）。

    Args:
        model: EditFlowsTransformer 模型
        data_encoder: SetEncoder 数据编码器
        x_values: 输入特征 (n_points, 3)
        y_target: 目标值 (n_points,)
        x0_token_ids: 初始 token 序列
        vocab: 词汇表
        scheduler: Kappa 调度器
        n_steps: 最大采样步数
        default_h: 默认步长
        device: 设备

    Returns:
        final_sequence: 最终采样的 token 序列
    """
    model.eval()
    data_encoder.eval()

    with torch.no_grad():
        # 准备条件编码
        encoder_input = prepare_encoder_input(x_values.unsqueeze(0), y_target.unsqueeze(0))
        condition = data_encoder(encoder_input)  # (1, num_features, hidden_dim)

        # 初始化序列
        batch_size = 1
        x_t = x0_token_ids.unsqueeze(0)  # (1, seq_len)
        seq_len = x_t.shape[1]

        # 初始化时间
        t = torch.zeros(batch_size, 1, device=device)

        step = 0
        # 当 t 接近 1 时继续运行，直到达到最大步数
        while step < n_steps:
            # 创建 padding mask
            x_pad_mask = (x_t == vocab.pad_token)

            # 前向传播获取速率和概率
            rates, ins_probs, sub_probs = model(
                tokens=x_t,
                time_step=t,
                padding_mask=x_pad_mask,
                condition=condition,
            )

            lambda_ins = rates[:, :, 0]  # (batch_size, seq_len)
            lambda_sub = rates[:, :, 1]
            lambda_del = rates[:, :, 2]

            if step == 0:
                print(f"  Initial lambda_ins: {lambda_ins[0, :5].tolist()}")
                print(f"  Initial lambda_sub: {lambda_sub[0, :5].tolist()}")
                print(f"  Initial lambda_del: {lambda_del[0, :5].tolist()}")

            # 自适应步长
            adapt_h = get_adaptive_h(default_h, t, scheduler)

            # 采样编辑操作（基于 Poisson 过程）
            non_pad_mask = ~x_pad_mask

            # 插入概率
            p_ins = 1 - torch.exp(-adapt_h * lambda_ins)
            ins_mask = (torch.rand_like(lambda_ins) < p_ins) & non_pad_mask

            # 删除/替换概率
            p_del_sub = 1 - torch.exp(-adapt_h * (lambda_sub + lambda_del))
            del_sub_mask = (torch.rand_like(lambda_sub) < p_del_sub) & non_pad_mask

            # 在需要删除/替换的位置，决定是删除还是替换
            p_del_given_del_sub = lambda_del / (lambda_del + lambda_sub + 1e-8)
            del_mask = del_sub_mask & (torch.rand_like(lambda_del) < p_del_given_del_sub)
            sub_mask = del_sub_mask & ~del_mask

            if step == 0:
                print(f"  adapt_h: {adapt_h.item():.4f}")
                print(f"  p_ins range: {p_ins[0, :5].tolist()}")
                print(f"  ins_mask sum: {ins_mask.sum().item()}")
                print(f"  del_mask sum: {del_mask.sum().item()}")
                print(f"  sub_mask sum: {sub_mask.sum().item()}")

            # 应用操作（内部会从概率分布采样 token）
            x_t, n_ins, n_del, n_sub = apply_ins_del_operations(
                x_t, ins_mask, del_mask, sub_mask,
                ins_probs, sub_probs,
                vocab.pad_token,
            )

            # 更新时间
            t = t + adapt_h
            step += 1

            if step % 10 == 0:
                print(f"  Step {step}: t={t.max().item():.4f}, Edits: {n_ins} ins, {n_del} del, {n_sub} sub")

    return x_t.squeeze(0)


def decode_sequence_to_expr(token_ids: torch.Tensor, vocab: Vocabulary) -> str:
    """将 token ID 序列解码为表达式字符串。

    Args:
        token_ids: token ID 序列
        vocab: 词汇表

    Returns:
        表达式字符串
    """
    # 移除 padding token
    tokens = []
    for token_id in token_ids:
        if token_id.item() == vocab.pad_token:
            continue
        try:
            token = vocab.id_to_token(token_id.item())
            tokens.append(token)
        except (IndexError, KeyError):
            continue

    # 简单转换为表达式（更复杂的转换可能需要后处理）
    expr = " ".join(tokens)
    return expr


def main():
    """命令行入口函数。"""
    parser = argparse.ArgumentParser(description="Edit Flow 采样推理")
    parser.add_argument("--data-path", type=str, default="data/test_sample.npz",
                        help="测试数据 NPZ 文件路径")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="要推理的样本索引")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/best_model",
                        help="模型检查点目录")
    parser.add_argument("--n-steps", type=int, default=1000,
                        help="最大采样步数")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")

    args = parser.parse_args()

    device = torch.device(args.device)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Edit Flow 采样推理 ===")
    print(f"数据路径: {args.data_path}")
    print(f"样本索引: {args.sample_idx}")
    print(f"检查点目录: {args.checkpoint_dir}")
    print(f"设备: {device}")

    # 加载模型
    print("\n加载模型...")
    model, data_encoder, vocab = CheckpointManager.load_model_only(args.checkpoint_dir, device)
    print(f"  词汇表大小: {vocab.vocab_size}")
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 加载数据
    print(f"\n加载数据 (样本 {args.sample_idx})...")
    x_values, y_target, x0_token_ids = load_test_sample(args.data_path, args.sample_idx, device)
    print(f"  x_values shape: {x_values.shape}")
    print(f"  y_target shape: {y_target.shape}")
    print(f"  初始序列长度: {len(x0_token_ids)}")
    print(f"  初始序列: {x0_token_ids.tolist()}")

    # 创建调度器
    scheduler = CubicScheduler(a=2.0, b=0.5)

    # 执行采样
    print(f"\n开始采样 (最大 {args.n_steps} 步)...")
    final_sequence = edit_flow_sampling(
        model=model,
        data_encoder=data_encoder,
        x_values=x_values,
        y_target=y_target,
        x0_token_ids=x0_token_ids,
        vocab=vocab,
        scheduler=scheduler,
        n_steps=args.n_steps,
        device=device,
    )

    # 解码结果
    print("\n=== 采样结果 ===")
    final_expr = decode_sequence_to_expr(final_sequence, vocab)
    print(f"最终序列长度: {len(final_sequence)}")
    print(f"最终序列: {final_sequence.tolist()}")
    print(f"表达式: {final_expr}")

    # 保存结果
    output_file = output_dir / f"sample_{args.sample_idx}_result.txt"
    with open(output_file, "w") as f:
        f.write(f"Sample Index: {args.sample_idx}\n")
        f.write(f"Initial Sequence: {x0_token_ids.tolist()}\n")
        f.write(f"Final Sequence: {final_sequence.tolist()}\n")
        f.write(f"Final Sequence Length: {len(final_sequence)}\n")
        f.write(f"Expression: {final_expr}\n")
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
