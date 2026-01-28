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
) -> float:
    """Single training epoch loop.

    Args:
        model: EditFlowsTransformer model
        data_loader: Data loader for symbolic regression
        optimizer: Optimizer (e.g., Adam)
        scheduler: Kappa scheduler for flow interpolation
        device: Device to train on
        vocab: Vocabulary for token IDs and vocab_size
        data_encoder: SetEncoder for encoding (x_values, y_target) pairs

    Returns:
        Average loss over the epoch
    """
    model.train()
    if data_encoder is not None:
        data_encoder.train()

    total_loss = 0.0
    num_batches = len(data_loader)

    for step, batch in enumerate(data_loader):
        x_0, x_1, z_0, z_1, t, x_values, y_target = batch

        condition = None
        if data_encoder is not None:
            encoder_input = prepare_encoder_input(x_values, y_target)
            condition = data_encoder(encoder_input)

        z_t = sample_cond_pt(x2prob(z_0, vocab.vocab_size), x2prob(z_1, vocab.vocab_size), t, scheduler)
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, vocab.pad_token, vocab.gap_token)
        uz_mask = make_ut_mask_from_z(
            z_t, z_1,
            vocab_size=vocab.vocab_size,
            pad_token=vocab.pad_token,
            gap_token=vocab.gap_token,
        )

        u_t, ins_probs, sub_probs = model.forward(
            tokens=x_t,
            time_step=t,
            padding_mask=x_pad_mask,
            condition=condition,
        )

        lambda_ins = u_t[:, :, 0]
        lambda_sub = u_t[:, :, 1]
        lambda_del = u_t[:, :, 2]

        u_tia_ins = lambda_ins.unsqueeze(-1) * ins_probs
        u_tia_sub = lambda_sub.unsqueeze(-1) * sub_probs
        u_tia_del = lambda_del.unsqueeze(-1)

        ux_cat = torch.cat([u_tia_ins, u_tia_sub, u_tia_del], dim=-1)
        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap_mask, z_pad_mask)

        # 数值稳定性修复：防止 t 接近 1 时除以零导致溢出
        kappa_t = scheduler(t)
        denom = (1 - kappa_t).clamp(min=1e-6)
        sched_coeff = (scheduler.derivative(t) / denom).to(device)
        sched_coeff = sched_coeff.clamp(max=50.0)

        uz_cat_safe = uz_cat.clamp(min=1e-8)
        log_uz_cat = uz_cat_safe.log().clamp(min=-20, max=20)
        u_tot = u_t.sum(dim=(1, 2))
        loss = u_tot - (log_uz_cat * uz_mask.to(device) * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        optimizer.step()

        # 输出训练信息
        print(f"  Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Grad Norm: {total_norm:.4f}")

        total_loss += loss.item()

    return total_loss / num_batches
