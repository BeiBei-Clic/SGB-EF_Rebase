"""Training script for Edit Flows model."""

import argparse
import numpy as np
import torch
from pathlib import Path
import pysnooper
from torch.utils.data import random_split

from src.data_loader.data_loader import SRDataLoader, SymbolicRegressionDataset, make_batch_cpu
from src.model.EditFlowsTransformer import EditFlowsTransformer
from src.model.data_embedding import SetEncoder
from src.model.vocab import Vocabulary
from src.core.training import train_one_epoch, evaluate_one_epoch
from src.core.flow_helper import CubicScheduler
from src.utils.checkpoint import CheckpointManager, TrainingState
from src.utils.lr_scheduler import create_warmup_cosine_scheduler
from src.data_loader.accelerate_loader import AccelerateSRDataLoader
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator

# 在装饰器生效前创建日志目录
Path("logs").mkdir(parents=True, exist_ok=True)


class _CollateFn:
    """可序列化的 collate_fn 包装器。"""
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        return make_batch_cpu(batch, self.vocab)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Edit Flows model")

    # Data
    parser.add_argument("--data-path", type=str, default="data/test_sample.npz",
                        help="Path to training data")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--lr-step-per-batch", action="store_true",
                        help="Step LR scheduler per batch instead of per epoch")

    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from checkpoint path")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--max-seq-len", type=int, default=48, help="Maximum sequence length")

    # Encoder
    parser.add_argument("--encoder-hidden", type=int, default=1024, help="Encoder hidden dimension")
    parser.add_argument("--encoder-heads", type=int, default=16, help="Encoder attention heads")
    parser.add_argument("--encoder-layers", type=int, default=12, help="Number of encoder layers")
    parser.add_argument("--encoder-inds", type=int, default=128, help="Number of inducing points")
    parser.add_argument("--encoder-input-normalization", action="store_true", default=True,
                        help="Normalize y_target in SetEncoder input")

    # Scheduler
    parser.add_argument("--scheduler-a", type=float, default=1.0, help="Scheduler parameter a")
    parser.add_argument("--scheduler-b", type=float, default=1.0, help="Scheduler parameter b")

    # Data split
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--disable-validation", action="store_true", help="Disable validation")

    return parser.parse_args()


@pysnooper.snoop('logs/debug.log')
def main():
    # 清理并重新创建日志文件
    log_path = Path("logs/debug.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    args = parse_args()

    # 在 Accelerator 初始化之前设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Accelerator 初始化
    accelerator = Accelerator(
        mixed_precision='no',
        gradient_accumulation_steps=1,
    )

    # Setup
    data_path = Path(args.data_path)

    if not data_path.exists():
        if accelerator.is_main_process:
            print(f"Data file not found: {data_path}")
            print("Please generate the data first")
        return

    # 仅主进程创建目录
    if accelerator.is_main_process:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)

    # 获取 input_dim（直接从 dataset 读取）
    dataset = SymbolicRegressionDataset(str(data_path), shuffle=False)
    input_dim = dataset[0]['input_dimension']
    if accelerator.is_main_process:
        print(f"Input dimension: {input_dim}")

    # Vocab
    vocab = Vocabulary(num_variables=input_dim)
    if accelerator.is_main_process:
        print(f"Vocabulary size: {vocab.vocab_size}")
        print(f"Gap token ID: {vocab.gap_token}")
        print(f"Pad token ID: {vocab.pad_token}")
        print(f"BOS token ID: {vocab.token_to_id('<s>')}")

    # 数据集划分
    total_size = len(dataset)
    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    if accelerator.is_main_process:
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")

    # 数据加载器
    num_workers = 0 if accelerator.num_processes > 1 else 4
    collate_fn = _CollateFn(vocab)

    # 训练集数据加载器
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if accelerator.num_processes > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 验证集数据加载器
    val_loader = None
    if val_size > 0 and not args.disable_validation:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if accelerator.num_processes > 1 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # 测试集数据加载器
    test_loader = None
    if test_size > 0:
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if accelerator.num_processes > 1 else None
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

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
        input_normalization=args.encoder_input_normalization,
    )
    if accelerator.is_main_process:
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
    )
    if accelerator.is_main_process:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer & Schedulers
    all_params = list(model.parameters()) + list(data_encoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    flow_scheduler = CubicScheduler(a=args.scheduler_a, b=args.scheduler_b)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch if args.lr_step_per_batch else args.epochs
    warmup_steps = args.warmup_epochs * steps_per_epoch if args.lr_step_per_batch else args.warmup_epochs
    lr_scheduler = create_warmup_cosine_scheduler(
        optimizer, total_steps, warmup_steps, args.min_lr
    )

    # 使用 prepare() 包装（在从检查点恢复之前）
    model, data_encoder, optimizer, lr_scheduler = accelerator.prepare(
        model, data_encoder, optimizer, lr_scheduler
    )

    # 从检查点恢复（需要在 prepare 之后使用 unwrap_model）
    if args.resume_from:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from}")
        training_state = CheckpointManager.load_checkpoint(
            args.resume_from, model, data_encoder, optimizer
        )
        start_epoch = training_state.epoch + 1
        if accelerator.is_main_process:
            print(f"Resumed from epoch {training_state.epoch}, best_loss: {training_state.best_loss:.4f}")
    else:
        training_state = TrainingState(epoch=0, global_step=0, best_loss=float('inf'))
        start_epoch = 0

    # Train
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # 训练一个 epoch
        avg_train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            flow_scheduler,
            vocab,
            data_encoder,
            accelerator,
            lr_scheduler=lr_scheduler,
            lr_step_per_batch=args.lr_step_per_batch,
        )
        if not args.lr_step_per_batch:
            lr_scheduler.step()
        if accelerator.is_main_process:
            print(f"Train loss: {avg_train_loss:.4f}")

        training_state.global_step += len(train_loader)
        training_state.epoch = epoch

        # 验证（如果启用）
        val_loss = None
        if val_loader is not None:
            if val_sampler:
                val_sampler.set_epoch(epoch)
            val_loss = evaluate_one_epoch(
                model, val_loader, flow_scheduler, vocab, data_encoder, accelerator
            )
            if accelerator.is_main_process:
                print(f"Val loss: {val_loss:.4f}")

        # 根据验证损失更新最佳模型（如果有验证集），
        # 否则根据训练损失
        current_loss = val_loss if val_loss is not None else avg_train_loss
        if current_loss < training_state.best_loss:
            training_state.best_loss = current_loss
            if accelerator.is_main_process:
                print(f"New best loss: {training_state.best_loss:.4f}")
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_encoder = accelerator.unwrap_model(data_encoder)
                CheckpointManager.save_model_only(
                    checkpoint_dir / "best_model", unwrapped_model, unwrapped_encoder, vocab
                )

        # 定期保存检查点
        if (epoch + 1) % args.checkpoint_every == 0:
            if accelerator.is_main_process:
                ckpt_path = checkpoint_dir / f"checkpoint-epoch-{epoch+1}"
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_encoder = accelerator.unwrap_model(data_encoder)
                CheckpointManager.save_checkpoint(
                    ckpt_path, unwrapped_model, unwrapped_encoder, optimizer, vocab, training_state
                )
                print(f"Checkpoint saved to {ckpt_path}")

    # 训练结束后在测试集上评估
    if test_loader is not None:
        if accelerator.is_main_process:
            print("\n--- Testing ---")
        test_loss = evaluate_one_epoch(
            model, test_loader, flow_scheduler, vocab, data_encoder, accelerator
        )
        if accelerator.is_main_process:
            print(f"Test loss: {test_loss:.4f}")

    # 训练结束后保存最终模型
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_encoder = accelerator.unwrap_model(data_encoder)
        CheckpointManager.save_model_only(
            checkpoint_dir / "final_model", unwrapped_model, unwrapped_encoder, vocab
        )
        print(f"\nFinal model saved to {checkpoint_dir / 'final_model'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
