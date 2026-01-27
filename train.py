"""Training script for Edit Flows model."""

import argparse
import torch
from pathlib import Path
import pysnooper

from src.data_loader.data_loader import SRDataLoader, SymbolicRegressionDataset
from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.data_embedding import SetEncoder
from src.model.vocab import Vocabulary
from src.core.training import train_one_epoch
from src.core.flow_helper import CubicScheduler
from src.utils.checkpoint import CheckpointManager, TrainingState
from src.utils.lr_scheduler import create_warmup_cosine_scheduler

# 在装饰器生效前创建日志目录
Path("logs").mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Edit Flows model")

    # Data
    parser.add_argument("--data-path", type=str, default="data/test_sample.npz",
                        help="Path to training data")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Number of warmup epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")

    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from checkpoint path")

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


# @pysnooper.snoop('logs/debug.log')
def main():
    args = parse_args()

    # Setup
    data_path = Path(args.data_path)

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please generate the data first")
        return

    # 检查点目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取 input_dim（直接从 dataset 读取）
    dataset = SymbolicRegressionDataset(str(data_path), shuffle=False)
    input_dim = dataset[0]['input_dimension']
    print(f"Input dimension: {input_dim}")

    # Vocab
    vocab = Vocabulary(num_variables=input_dim)
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Gap token ID: {vocab.gap_token}")
    print(f"Pad token ID: {vocab.pad_token}")
    print(f"BOS token ID: {vocab.token_to_id('<s>')}")

    # 数据加载（使用 vocab）
    data_loader = SRDataLoader(str(data_path), batch_size=args.batch_size, shuffle=True, device=device, vocab=vocab)
    print(f"Dataset size: {len(data_loader.dataset)}")
    print(f"Number of batches: {len(data_loader)}")

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

    # Optimizer & Schedulers
    all_params = list(model.parameters()) + list(data_encoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    flow_scheduler = CubicScheduler(a=args.scheduler_a, b=args.scheduler_b)
    lr_scheduler = create_warmup_cosine_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.min_lr
    )

    # 从检查点恢复
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        training_state = CheckpointManager.load_checkpoint(
            args.resume_from, model, data_encoder, optimizer, device
        )
        start_epoch = training_state.epoch + 1
        print(f"Resumed from epoch {training_state.epoch}, best_loss: {training_state.best_loss:.4f}")
    else:
        training_state = TrainingState(epoch=0, global_step=0, best_loss=float('inf'))
        start_epoch = 0

    # Train
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        avg_loss = train_one_epoch(model, data_loader, optimizer, flow_scheduler, device, vocab, data_encoder)
        lr_scheduler.step()
        print(f"Average loss: {avg_loss:.4f}")

        training_state.global_step += len(data_loader)
        training_state.epoch = epoch

        # 更新最佳损失并保存最佳模型
        if avg_loss < training_state.best_loss:
            training_state.best_loss = avg_loss
            print(f"New best loss: {training_state.best_loss:.4f}")
            CheckpointManager.save_model_only(
                checkpoint_dir / "best_model", model, data_encoder, vocab
            )

        # 定期保存检查点
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint-epoch-{epoch+1}"
            CheckpointManager.save_checkpoint(
                ckpt_path, model, data_encoder, optimizer, vocab, training_state
            )
            print(f"Checkpoint saved to {ckpt_path}")

    # 训练结束后保存最终模型
    CheckpointManager.save_model_only(
        checkpoint_dir / "final_model", model, data_encoder, vocab
    )
    print(f"\nFinal model saved to {checkpoint_dir / 'final_model'}")


if __name__ == "__main__":
    main()
