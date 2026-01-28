"""Accelerate-compatible data loader for symbolic regression."""

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.data_loader.data_loader import SymbolicRegressionDataset, make_batch_cpu


class _CollateFn:
    """可序列化的 collate_fn 包装器。"""
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        return make_batch_cpu(batch, self.vocab)


class AccelerateSRDataLoader:
    """Data loader wrapper compatible with Accelerate distributed training."""

    def __init__(
        self,
        npz_path: str,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        vocab,
    ):
        # 分布式训练时禁用 num_workers，避免创建过多子进程导致 OOM
        if dist.is_initialized() and dist.get_world_size() > 1:
            num_workers = 0

        self.dataset = SymbolicRegressionDataset(npz_path, shuffle=shuffle)
        self.sampler = DistributedSampler(self.dataset, shuffle=shuffle) if shuffle else None
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            shuffle=shuffle and self.sampler is None,
            num_workers=num_workers,
            collate_fn=_CollateFn(vocab),
            pin_memory=True,
        )

    def set_epoch(self, epoch: int):
        if self.sampler:
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
