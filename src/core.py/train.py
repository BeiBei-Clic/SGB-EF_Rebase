"""Training loop for Edit Flows model.

Based on edit-flows-demo/main.py structure, adapted for symbolic regression.
"""

import torch

from src.data_loader.data_loader import SRDataLoader, make_batch
from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.data_embedding import SetEncoder, prepare_encoder_input
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
        z_t = sample_cond_pt(x2prob(z_0, vocab.data_vocab_size), x2prob(z_1, vocab.data_vocab_size), t, scheduler)

        # 4. 计算 x_t（移除 gap tokens）
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, vocab.pad_token, vocab.gap_token)

        # 5. 计算编辑操作掩码
        uz_mask = make_ut_mask_from_z(
            z_t,
            z_1,
            vocab_size=vocab.data_vocab_size,
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

    # 获取数据维度信息
    first_batch = next(iter(data_loader))
    _, _, _, _, _, x_values, y_target = first_batch
    input_dim = x_values.shape[-1]  # x_values 维度
    print(f"Input dimension: {input_dim}")

    # 创建 SetEncoder（数据编码器）
    data_encoder = SetEncoder(
        dim_input=input_dim + 1,  # x_values + y_target
        dim_hidden=128,
        num_heads=4,
        num_inds=32,
        ln=True,
        n_l_enc=2,
        num_features=1,
        linear=True,
    ).to(device)
    print(f"Data encoder parameters: {sum(p.numel() for p in data_encoder.parameters() if p.requires_grad)}")

    # Create model with cross-attention enabled
    model = EditFlowsTransformer(
        vocab_size=vocab.data_vocab_size,  # Use data vocab size for compatibility
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        bos_token_id=data_bos_token_id,
        pad_token_id=data_pad_token_id,
        num_cond_features=1,  # 与 SetEncoder 的 num_features 对应
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create optimizer - 包含 encoder 和 model 参数
    all_params = list(model.parameters()) + list(data_encoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.0001)

    # Create scheduler
    scheduler = CubicScheduler(a=1.0, b=1.0)

    train_one_epoch(model, data_loader, optimizer, scheduler, device, vocab, data_encoder)


if __name__ == "__main__":
    main()
