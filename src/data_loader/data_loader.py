"""Data loader for symbolic regression with Edit Flows."""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# Special token IDs based on data format analysis
# These are derived from the actual data files
GAP_TOKEN_ID = 0   # Gap token for alignment
PAD_TOKEN_ID = 4   # Padding token
BOS_TOKEN_ID = 2   # Beginning of sequence (used as length prefix in data)


class SymbolicRegressionDataset(Dataset):
    """Dataset for symbolic regression data from NPZ files."""

    def __init__(self, npz_path: str, shuffle: bool = True):
        """Load data from NPZ file."""
        self.data = np.load(npz_path, allow_pickle=True)
        self.num_samples = len(self.data['input_dimensions'])
        self.indices = np.random.permutation(self.num_samples) if shuffle else np.arange(self.num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        idx = self.indices[idx]
        return {
            'input_dimension': int(self.data['input_dimensions'][idx]),
            'x_values': torch.from_numpy(self.data['x_values'][idx]).float(),
            'y_target': torch.from_numpy(self.data['y_target'][idx]),
            'z0_token_ids': torch.from_numpy(self.data['z0_token_ids'][idx]).long(),
            'z1_token_ids': torch.from_numpy(self.data['z1_token_ids'][idx]).long(),
        }


def make_batch(
    dataset: SymbolicRegressionDataset,
    indices: List[int],
    device: torch.device = torch.device('cpu'),
) -> tuple:
    """Create a training batch from dataset indices.

    Returns:
        z_0: Aligned base sequences (batch_size, seq_len) - with gap tokens
        z_1: Aligned target sequences (batch_size, seq_len) - with gap tokens
        t: Time steps (batch_size, 1) - sampled from Uniform(0, 1)
        padding_mask: Boolean mask (batch_size, seq_len) - True where padded
        x_values: Regression input features (batch_size, n_points, input_dim)
        y_target: Regression target values (batch_size, n_points)
    """
    batch_size = len(indices)
    samples = [dataset[i] for i in indices]

    z0_ids_list = []
    z1_ids_list = []
    x_values_list = []
    y_target_list = []

    for sample in samples:
        # Process z0_token_ids: [length_prefix, token1, token2, ..., padding]
        z0_ids = sample['z0_token_ids'][1:]  # Remove length prefix
        pad_mask = z0_ids != PAD_TOKEN_ID
        if pad_mask.any():
            actual_len = pad_mask.nonzero(as_tuple=False)[-1].item() + 1
            z0_ids = z0_ids[:actual_len]
        z0_ids_list.append(z0_ids)

        # Process z1_token_ids
        z1_ids = sample['z1_token_ids'][1:]  # Remove length prefix
        pad_mask = z1_ids != PAD_TOKEN_ID
        if pad_mask.any():
            actual_len = pad_mask.nonzero(as_tuple=False)[-1].item() + 1
            z1_ids = z1_ids[:actual_len]
        z1_ids_list.append(z1_ids)

        x_values_list.append(sample['x_values'])
        y_target_list.append(sample['y_target'])

    # Pad sequences to max length
    z_max_len = max(max(len(z) for z in z0_ids_list), max(len(z) for z in z1_ids_list))

    z_0 = torch.stack([F.pad(z, (0, z_max_len - len(z)), value=PAD_TOKEN_ID) for z in z0_ids_list]).to(device)
    z_1 = torch.stack([F.pad(z, (0, z_max_len - len(z)), value=PAD_TOKEN_ID) for z in z1_ids_list]).to(device)

    t = torch.rand(batch_size, 1, device=device)
    padding_mask = (z_1 == PAD_TOKEN_ID)

    # Stack x_values and y_target into batch tensors
    x_values_batch = torch.stack(x_values_list).to(device)  # (batch_size, n_points, input_dim)
    y_target_batch = torch.stack(y_target_list).to(device)  # (batch_size, n_points)

    return z_0, z_1, t, padding_mask, x_values_batch, y_target_batch


class SRDataLoader:
    """Data loader wrapper for symbolic regression."""

    def __init__(
        self,
        npz_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        device: torch.device = torch.device('cpu'),
    ):
        self.dataset = SymbolicRegressionDataset(npz_path, shuffle=shuffle)
        self.batch_size = batch_size
        self.device = device
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield make_batch(self.dataset, batch_indices, self.device)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_data_loader(
    npz_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    device: torch.device = torch.device('cpu'),
) -> SRDataLoader:
    """Convenience function to create data loader."""
    return SRDataLoader(npz_path, batch_size, shuffle, device)
