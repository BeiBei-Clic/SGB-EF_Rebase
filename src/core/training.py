"""Core training loop for Edit Flows model.

Based on edit-flows-demo/main.py structure, adapted for symbolic regression.
"""

import torch

from src.data_loader.data_loader import SRDataLoader
from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.data_embedding import SetEncoder, prepare_encoder_input
from src.model.vocab import Vocabulary
from .flow_helper import (
    x2prob,
    sample_cond_pt,
    CubicScheduler,
    make_ut_mask_from_z,
    rm_gap_tokens,
    fill_gap_tokens_with_repeats,
)


def train_one_epoch(
    model: EditFlowsTransformer,
    data_loader: SRDataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CubicScheduler,
    device: torch.device,
    vocab: Vocabulary,
    data_encoder: SetEncoder = None,
):
    """Single training epoch loop.

    Args:
        model: EditFlowsTransformer model
        data_loader: Data loader for symbolic regression
        optimizer: Optimizer (e.g., Adam)
        scheduler: Kappa scheduler for flow interpolation
        device: Device to train on
        vocab: Vocabulary for token IDs and vocab_size
        data_encoder: SetEncoder for encoding (x_values, y_target) pairs
    """
    model.train()
    if data_encoder is not None:
        data_encoder.train()

    for step, batch in enumerate(data_loader):
        # 1. 读取 batch
        x_0, x_1, z_0, z_1, t, x_values, y_target = batch

        # 2. 编码数据点对作为条件
        condition = None
        if data_encoder is not None:
            encoder_input = prepare_encoder_input(x_values, y_target)  # (B, n, d+1)
            condition = data_encoder(encoder_input)  # (B, num_features, hidden_dim)

        # 3. 计算 z_t（在 Z 空间插值）
        z_t = sample_cond_pt(x2prob(z_0, vocab.vocab_size), x2prob(z_1, vocab.vocab_size), t, scheduler)

        # 4. 计算 x_t（移除 gap tokens）
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, vocab.pad_token, vocab.gap_token)

        # 5. 计算编辑操作掩码
        uz_mask = make_ut_mask_from_z(
            z_t,
            z_1,
            vocab_size=vocab.vocab_size,
            pad_token=vocab.pad_token,
            gap_token=vocab.gap_token,
        )

        # 6. 模型前向传播
        u_t, ins_probs, sub_probs = model.forward(
            tokens=x_t,
            time_step=t,
            padding_mask=x_pad_mask,
            condition=condition,
        )

        # 6. 准备损失计算的中间变量
        lambda_ins = u_t[:, :, 0]  # (batch_size, x_seq_len)
        lambda_sub = u_t[:, :, 1]  # (batch_size, x_seq_len)
        lambda_del = u_t[:, :, 2]  # (batch_size, x_seq_len)

        u_tia_ins = lambda_ins.unsqueeze(-1) * ins_probs  # (batch_size, x_seq_len, vocab_size)
        u_tia_sub = lambda_sub.unsqueeze(-1) * sub_probs  # (batch_size, x_seq_len, vocab_size)
        u_tia_del = lambda_del.unsqueeze(-1)  # (batch_size, x_seq_len, 1)

        ux_cat = torch.cat([u_tia_ins, u_tia_sub, u_tia_del], dim=-1)  # (batch_size, x_seq_len, 2 * vocab_size + 1)
        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap_mask, z_pad_mask)  # (batch_size, z_seq_len, 2 * vocab_size + 1)

        sched_coeff = (scheduler.derivative(t) / (1 - scheduler(t))).to(device)
        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)
        u_tot = u_t.sum(dim=(1, 2))
        loss = u_tot - (log_uz_cat * uz_mask.to(device) * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}/{len(data_loader)}: loss = {loss.item():.4f}")
