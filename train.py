"""Training script for Edit Flows model."""

import argparse
import torch
from pathlib import Path

from src.data_loader.data_loader import SRDataLoader
from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.data_embedding import SetEncoder
from src.model.vocab import Vocabulary
from src.core.training import train_one_epoch
from src.core.flow_helper import CubicScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train Edit Flows model")

    # Data
    parser.add_argument("--data-path", type=str, default="data/test_sample.npz",
                        help="Path to training data")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length")

    # Encoder
    parser.add_argument("--encoder-hidden", type=int, default=128, help="Encoder hidden dimension")
    parser.add_argument("--encoder-heads", type=int, default=4, help="Encoder attention heads")
    parser.add_argument("--encoder-layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--encoder-inds", type=int, default=32, help="Number of inducing points")

    # Scheduler
    parser.add_argument("--scheduler-a", type=float, default=1.0, help="Scheduler parameter a")
    parser.add_argument("--scheduler-b", type=float, default=1.0, help="Scheduler parameter b")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    data_path = Path(args.data_path)
    if not data_path.exists():
        data_path = Path("data/test_sample.npz")
        print(f"Training data not found, using test data: {data_path}")

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please generate the data first")
        return

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    data_loader = SRDataLoader(str(data_path), batch_size=args.batch_size, shuffle=True, device=device)
    print(f"Dataset size: {len(data_loader.dataset)}")
    print(f"Number of batches: {len(data_loader)}")

    # 获取 input_dim
    first_batch = next(iter(data_loader))
    _, _, _, _, _, x_values, _ = first_batch
    input_dim = x_values.shape[-1]
    print(f"Input dimension: {input_dim}")

    # Vocab
    vocab = Vocabulary(num_variables=input_dim)
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Gap token ID: {vocab.gap_token}")
    print(f"Pad token ID: {vocab.pad_token}")
    print(f"BOS token ID: {vocab.token_to_id('<s>')}")

    # Encoder
    data_encoder = SetEncoder(
        dim_input=input_dim + 1,
        dim_hidden=args.encoder_hidden,
        num_heads=args.encoder_heads,
        num_inds=args.encoder_inds,
        ln=True,
        n_l_enc=args.encoder_layers,
        num_features=1,
        linear=True,
    ).to(device)
    print(f"Data encoder parameters: {sum(p.numel() for p in data_encoder.parameters() if p.requires_grad)}")

    # Model
    model = EditFlowsTransformer(
        vocab_size=vocab.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        bos_token_id=vocab.token_to_id('<s>'),
        pad_token_id=vocab.pad_token,
        num_cond_features=1,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer & Scheduler
    all_params = list(model.parameters()) + list(data_encoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    scheduler = CubicScheduler(a=args.scheduler_a, b=args.scheduler_b)

    # Train
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_one_epoch(model, data_loader, optimizer, scheduler, device, vocab, data_encoder)


if __name__ == "__main__":
    main()
