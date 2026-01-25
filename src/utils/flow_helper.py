"""Flow helper functions for Edit Flows training.

Ported from edit-flows-demo with modifications to use Vocabulary class.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch


def x2prob(x: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Converts sequence of tokens to class distribution representation."""
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()


def sample_p(pt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Samples sequence from class distribution representation."""
    b, l, _ = pt.shape
    pt = pt.reshape(b * l, -1)
    xt = torch.multinomial(pt / temperature, 1)
    return xt.reshape(b, l)


def sample_cond_pt(
    p0: torch.Tensor,
    p1: torch.Tensor,
    t: torch.Tensor,
    kappa: "KappaScheduler",
) -> torch.Tensor:
    """Sample from conditional probability path at time t."""
    t = t.reshape(-1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt)


class KappaScheduler(ABC):
    """Base class for kappa schedulers."""

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CubicScheduler(KappaScheduler):
    """Cubic kappa scheduler: κ(t) = -2t³ + 3t² + a(t³ - 2t² + t) + b(t³ - t²)."""

    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return -2 * (t**3) + 3 * (t**2) + self.a * (t**3 - 2 * t**2 + t) + self.b * (t**3 - t**2)

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -6 * (t**2) + 6 * t + self.a * (3 * t**2 - 4 * t + 1) + self.b * (3 * t**2 - 2 * t)


def rm_gap_tokens(
    z: torch.Tensor,
    pad_token: int,
    gap_token: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove gap tokens from a batched tensor and right-pad with PAD_TOKEN.

    Returns:
        x: Tensor with gap tokens removed
        x_pad_mask: Padding mask for x (True where x == PAD_TOKEN)
        z_gap_mask: Mask for gap tokens in z (True where z == GAP_TOKEN)
        z_pad_mask: Padding mask for z (True where z == PAD_TOKEN)
    """
    import torch.nn.functional as F

    batch_size, _ = z.shape
    z_no_gap = []
    for b in range(batch_size):
        z_no_pad = z[b][z[b] != pad_token]
        z_no_gap.append(z_no_pad[z_no_pad != gap_token])
    max_len = max(len(z) for z in z_no_gap)
    x = torch.stack([F.pad(z, (0, max_len - len(z)), value=pad_token) for z in z_no_gap], dim=0).long()
    x_pad_mask = (x == pad_token)
    z_gap_mask = (z == gap_token)
    z_pad_mask = (z == pad_token)
    assert ((~x_pad_mask).sum(1) + z_gap_mask.sum(1)).equal((~z_pad_mask).sum(1))
    return x, x_pad_mask, z_gap_mask, z_pad_mask


def make_ut_mask_from_z(
    z_t: torch.Tensor,
    z_1: torch.Tensor,
    vocab_size: int,
    pad_token: int,
    gap_token: int,
) -> torch.Tensor:
    """Create a mask for u_cat based on differences between z_t and z_1.

    For each position i where z_t and z_1 differ, we index as follows:
    - z_t[i] = GAP_TOKEN & z_1[i] = c => u_mask[i, insert, c] = 1
    - z_t[i] = c & z_1[i] = GAP_TOKEN => u_mask[i, delete] = 1
    - z_t[i] = c1 & z_1[i] = c2 => u_mask[i, substitute, c1, c2] = 1

    Returns:
        u_mask: (batch_size, z_seq_len, 2 * vocab_size + 1, bool)
    """
    batch_size, z_seq_len = z_t.shape
    n_ops = 2 * vocab_size + 1  # insert + substitute + delete

    z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq  # (batch_size, z_seq_len)
    z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq  # (batch_size, z_seq_len)
    z_sub = z_neq & ~z_ins & ~z_del  # (batch_size, z_seq_len)

    # mask (batch_size, z_seq_len, u_ops) where 1 indicates operation that bring z_t closer to z_1
    u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    u_mask[z_ins, z_1[z_ins]] = True
    u_mask[z_sub, z_1[z_sub] + vocab_size] = True
    u_mask[:, :, -1][z_del] = True

    assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
    assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

    return u_mask


def fill_gap_tokens_with_repeats(
    x_ut: torch.Tensor,
    z_gap_mask: torch.Tensor,
    z_pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Fill gap token positions with repeated values from non-gap positions.

    Args:
        x_ut: (batch_size, x_seq_len, channels)
        z_gap_mask: (batch_size, z_seq_len)
        z_pad_mask: (batch_size, z_seq_len)

    Returns:
        result: (batch_size, z_seq_len, channels)
    """
    batch_size, _ = z_gap_mask.shape
    _, x_seq_len, _ = x_ut.shape

    # Use cumsum on non-gap positions to point to the last valid non-gap position
    non_gap_mask = ~z_gap_mask  # Invert mask to get non-gap positions
    indices = non_gap_mask.cumsum(dim=1) - 1  # (batch_size, z_seq_len)
    indices = indices.clamp(min=0, max=x_seq_len - 1)  # Ensure indices are within bounds

    # Use indices to gather from x_ut
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]  # (batch_size, z_seq_len, channels)
    result[z_pad_mask] = 0  # Set pad positions to 0
    return result
