"""Edit Flow 采样推理模块。

实现基于 Poisson 过程的编辑流采样，用于符号回归表达式生成。
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pysnooper

from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.data_embedding import SetEncoder, prepare_encoder_input
from src.model.vocab import Vocabulary
from src.utils.checkpoint import CheckpointManager
from src.core.flow_helper import CubicScheduler


def get_adaptive_h(default_h: float, t: torch.Tensor, scheduler: CubicScheduler) -> torch.Tensor:
    """计算自适应时间步长。

    根据调度器公式精确计算，确保时间步长不会导致 t 超过 1。

    Args:
        default_h: 默认步长
        t: 当前时间 (batch_size, 1)
        scheduler: Kappa 调度器

    Returns:
        adapt_h: 自适应步长 (batch_size, 1)
    """
    # 基于调度器公式计算理论上界
    coeff = (1 - scheduler(t)) / scheduler.derivative(t)
    _h = default_h * torch.ones_like(t)
    h_adapt = torch.minimum(_h, coeff)
    return h_adapt


def apply_ins_del_operations(
    x_t: torch.Tensor,
    ins_mask: torch.Tensor,
    del_mask: torch.Tensor,
    ins_tokens: torch.Tensor,
    pad_token_id: int,
    max_seq_len: int = 512,
) -> Tuple[torch.Tensor, int, int]:
    """应用插入、删除操作到序列。

    正确处理插入和删除操作，包括同时插入+删除的情况（作为替换）。

    Args:
        x_t: 当前序列 (batch_size, seq_len)
        ins_mask: 插入掩码 (batch_size, seq_len)
        del_mask: 删除掩码 (batch_size, seq_len)
        ins_tokens: 插入的 token (batch_size, seq_len)
        pad_token_id: 填充 token ID
        max_seq_len: 最大序列长度

    Returns:
        new_x_t: 操作后的序列
        n_ins: 插入操作数量
        n_del: 删除操作数量
    """
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # 处理同时插入+删除的情况 -> 替换
    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    n_replace = replace_mask.sum().item()
    if n_replace > 0:
        x_t_modified[replace_mask] = ins_tokens[replace_mask]

    # 更新 mask，排除已处理的替换
    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask

    # 计算新序列长度
    xt_pad_mask = (x_t == pad_token_id)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())

    if max_new_len <= 0:
        return (
            torch.full((batch_size, 1), pad_token_id, dtype=x_t.dtype, device=device),
            eff_ins_mask.sum().item(),
            eff_del_mask.sum().item(),
        )

    # 预分配结果
    x_new = torch.full((batch_size, max_new_len), pad_token_id, dtype=x_t.dtype, device=device)

    # 计算位置映射
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    cum_del = torch.cumsum(eff_del_mask.float(), dim=1)
    cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)
    cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)

    # 放置未删除的 token
    new_pos = pos_idx + cum_ins_before - cum_del
    keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)
    if keep_mask.any():
        x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]

    # 放置插入的 token（在对应位置之后插入）
    if eff_ins_mask.any():
        ins_pos = new_pos + 1
        ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)
        if ins_valid.any():
            x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[ins_valid]

    # 限制最大长度
    if max_new_len > max_seq_len:
        max_new_len = max_seq_len
        x_new = x_new[:, :max_new_len]

    return x_new, eff_ins_mask.sum().item(), eff_del_mask.sum().item()


def load_test_sample(npz_path: str, sample_idx: int, device: torch.device, vocab: Vocabulary = None) -> Tuple:
    """从 NPZ 文件加载指定样本。

    Args:
        npz_path: NPZ 文件路径
        sample_idx: 样本索引
        device: 设备

    Returns:
        x_values: 输入特征 (n_points, 3)
        y_target: 目标值 (n_points,)
        x0_token_ids: 初始 token 序列
        x1_token_ids: 目标 token 序列
    """
    data = np.load(npz_path, allow_pickle=True)

    x_values = torch.from_numpy(data['x_values'][sample_idx]).float().to(device)
    y_target = torch.from_numpy(data['y_target'][sample_idx]).float().to(device)
    x0_token_ids = torch.from_numpy(data['x0_token_ids'][sample_idx]).long().to(device)
    x1_token_ids = torch.from_numpy(data['x1_token_ids'][sample_idx]).long().to(device)

    # 与训练时保持一致，在开头添加 <s> token
    if vocab is not None:
        bos_token_id = vocab.token_to_id('<s>')
        if x0_token_ids[0] != bos_token_id:
            x0_token_ids = F.pad(x0_token_ids, (1, 0), value=bos_token_id)

    return x_values, y_target, x0_token_ids, x1_token_ids

@pysnooper.snoop('logs/inference.log')
def edit_flow_sampling(
    model: EditFlowsTransformer,
    data_encoder: SetEncoder,
    x_values: torch.Tensor,
    y_target: torch.Tensor,
    x0_token_ids: torch.Tensor,
    vocab: Vocabulary,
    scheduler: CubicScheduler,
    n_steps: int = 10,
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
        while step < n_steps and t.max() <= 1 - default_h:
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

            # 归一化：每个位置的三个操作概率和为 1
            total = lambda_ins + lambda_sub + lambda_del + 1e-8
            lambda_ins = lambda_ins / total
            lambda_sub = lambda_sub / total
            lambda_del = lambda_del / total

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

            # 先采样 token
            ins_tokens = torch.full_like(x_t, vocab.pad_token, dtype=torch.long)
            sub_tokens = torch.full_like(x_t, vocab.pad_token, dtype=torch.long)

            if non_pad_mask.any():
                ins_tokens[non_pad_mask] = torch.multinomial(ins_probs[non_pad_mask], 1).squeeze(-1)
                sub_tokens[non_pad_mask] = torch.multinomial(sub_probs[non_pad_mask], 1).squeeze(-1)

            # 应用替换操作到原序列
            n_sub = sub_mask.sum().item()
            x_t[sub_mask] = sub_tokens[sub_mask]

            # 应用插入/删除操作
            x_t, n_ins, n_del = apply_ins_del_operations(
                x_t, ins_mask, del_mask,
                ins_tokens,
                vocab.pad_token,
                max_seq_len=x_t.shape[1] + 64,
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
    # 清理并重新创建日志文件
    log_path = Path("logs/inference.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

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
    x_values, y_target, x0_token_ids, x1_token_ids = load_test_sample(args.data_path, args.sample_idx, device, vocab)
    print(f"  x_values shape: {x_values.shape}")
    print(f"  y_target shape: {y_target.shape}")

    # 解码并打印初始表达式和目标表达式
    bos_id = vocab.token_to_id('<s>')
    x0_print = x0_token_ids[x0_token_ids != bos_id]
    x1_print = x1_token_ids[x1_token_ids != bos_id]
    initial_expr = decode_sequence_to_expr(x0_print, vocab)
    target_expr = decode_sequence_to_expr(x1_print, vocab)
    print(f"  初始表达式: {initial_expr}")
    print(f"  目标表达式: {target_expr}")
    print(f"  初始序列: {x0_print.tolist()}")
    print(f"  目标序列: {x1_print.tolist()}")

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
    print(f"最终表达式: {final_expr}")
    print(f"最终序列: {final_sequence.tolist()}")

    # 保存结果
    output_file = output_dir / f"sample_{args.sample_idx}_result.txt"
    with open(output_file, "w") as f:
        f.write(f"Sample Index: {args.sample_idx}\n")
        f.write(f"Initial Expression: {initial_expr}\n")
        f.write(f"Target Expression: {target_expr}\n")
        f.write(f"Final Expression: {final_expr}\n")
        f.write(f"\nInitial Sequence: {x0_token_ids.tolist()}\n")
        f.write(f"Target Sequence: {x1_token_ids.tolist()}\n")
        f.write(f"Final Sequence: {final_sequence.tolist()}\n")
        f.write(f"Final Sequence Length: {len(final_sequence)}\n")
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
