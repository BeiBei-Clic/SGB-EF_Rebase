"""Training loop for Edit Flows model.

Based on edit-flows-demo/main.py structure, adapted for symbolic regression.
"""

import torch

from src.data_loader.data_loader import SRDataLoader, make_batch
from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.vocab import Vocabulary
from src.utils.flow_helper import (
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
    debug: bool = False,
):
    """Single training epoch loop.

    Args:
        model: EditFlowsTransformer model
        data_loader: Data loader for symbolic regression
        optimizer: Optimizer (e.g., Adam)
        scheduler: Kappa scheduler for flow interpolation
        device: Device to train on
        vocab: Vocabulary for token IDs and vocab_size
        debug: If True, print debug information
    """
    model.train()

    for step, batch in enumerate(data_loader):
        # 1. 读取 batch
        x_0, x_1, z_0, z_1, t, x_values, y_target = batch

        # 2. 计算 z_t（在 Z 空间插值）
        z_t = sample_cond_pt(x2prob(z_0, vocab.data_vocab_size), x2prob(z_1, vocab.data_vocab_size), t, scheduler)

        # 3. 计算 x_t（移除 gap tokens）
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, vocab.pad_token, vocab.gap_token)

        # 4. 计算编辑操作掩码
        uz_mask = make_ut_mask_from_z(
            z_t,
            z_1,
            vocab_size=vocab.data_vocab_size,
            pad_token=vocab.pad_token,
            gap_token=vocab.gap_token,
        )

        # 5. 模型前向传播
        u_t, ins_probs, sub_probs = model.forward(
            tokens=x_t,
            time_step=t,
            padding_mask=x_pad_mask,
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

        # Debug output
        if debug and step == 0:
            print(f"\n=== Step {step} Debug ===")
            print(f"x_0 shape: {x_0.shape}, x_1 shape: {x_1.shape}")
            print(f"z_0 shape: {z_0.shape}, z_1 shape: {z_1.shape}")
            print(f"z_t shape: {z_t.shape}, x_t shape: {x_t.shape}")
            print(f"t shape: {t.shape}, t[0]: {t[0, 0].item():.4f}")
            print(f"uz_mask shape: {uz_mask.shape}, num_edits: {uz_mask.sum().item()}")
            print(f"u_t shape: {u_t.shape}")
            print(f"ux_cat shape: {ux_cat.shape}, uz_cat shape: {uz_cat.shape}")
            print(f"lambda_ins mean: {lambda_ins.mean().item():.4f}")
            print(f"lambda_sub mean: {lambda_sub.mean().item():.4f}")
            print(f"lambda_del mean: {lambda_del.mean().item():.4f}")

        # TODO: 损失计算
        # sched_coeff = (scheduler.derivative(t) / (1 - scheduler(t))).to(device)
        # log_uz_cat = torch.clamp(uz_cat.log(), min=-20)
        # u_tot = u_t.sum(dim=(1, 2))
        # loss = u_tot - (log_uz_cat * uz_mask.to(device) * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        # loss = loss.mean()

        # TODO: 反向传播和优化器更新
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # TODO: 记录指标
        # metrics["loss"].append(loss.item())
        # metrics["u_ins"].append(lambda_ins.sum(dim=1).mean().item())
        # metrics["u_sub"].append(lambda_sub.sum(dim=1).mean().item())
        # metrics["u_del"].append(lambda_del.sum(dim=1).mean().item())

        if debug and step == 0:
            break


def main():
    """Main training function for testing the training loop."""
    # Setup
    from pathlib import Path

    # Try to use training data, fallback to test data
    data_path = Path("data/sr_train_data.npz")
    if not data_path.exists():
        data_path = Path("data/test_sample.npz")
        print(f"Training data not found, using test data: {data_path}")

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please generate the data first")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create vocabulary - data was generated with different vocab size
    # For now, use a larger vocab_size to accommodate the data
    # TODO: Fix token mapping consistency
    vocab = Vocabulary(num_variables=1, data_vocab_size=25)  # For single variable x0

    # Data uses hardcoded token IDs - need these for model to match data format
    data_bos_token_id = 2
    data_pad_token_id = 4

    print(f"Vocabulary size: {vocab.vocab_size} (model will use {vocab.data_vocab_size})")
    print(f"Gap token ID: {vocab.gap_token}")
    print(f"Pad token ID: {vocab.pad_token}")
    print(f"Data BOS token ID: {data_bos_token_id}")
    print(f"Data PAD token ID: {data_pad_token_id}")

    # Create data loader
    data_loader = SRDataLoader(
        str(data_path),
        batch_size=2,
        shuffle=True,
        device=device,
    )
    print(f"Dataset size: {len(data_loader.dataset)}")
    print(f"Number of batches: {len(data_loader)}")

    # Create model
    model = EditFlowsTransformer(
        vocab_size=vocab.data_vocab_size,  # Use data vocab size for compatibility
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        bos_token_id=data_bos_token_id,
        pad_token_id=data_pad_token_id,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Create scheduler
    scheduler = CubicScheduler(a=1.0, b=1.0)

    # Run one training step with debug enabled
    print("\n=== Running training loop with debug ===")
    train_one_epoch(model, data_loader, optimizer, scheduler, device, vocab, debug=True)
    print("\n=== Training loop completed successfully ===")


if __name__ == "__main__":
    main()
