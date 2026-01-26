"""Edit path creation for sequence alignment."""

from typing import List, Tuple
from src.model.vocab import Vocabulary


def create_edit_path(
    target_token_ids: List[int],
    base_token_ids: List[int] = None,
    vocab: Vocabulary = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """创建编辑路径（z0, z1, x0, x1）.

    Args:
        target_token_ids: 目标表达式 token ids
        base_token_ids: 基础表达式 token ids（默认为空）
        vocab: 词表对象

    Returns:
        z0_ids: 对齐的 base 序列 token ids（含 gap token）
        z1_ids: 对齐的 target 序列 token ids（含 gap token）
        x0_ids: 非对齐的 base 序列 token ids（无 gap token）
        x1_ids: 非对齐的 target 序列 token ids（无 gap token）
    """
    if base_token_ids is None:
        base_token_ids = []

    if vocab is None:
        vocab = Vocabulary()

    gap_token_id = vocab.gap_token

    # 非对齐序列（无 gap）
    x0_ids = base_token_ids.copy()
    x1_ids = target_token_ids.copy()

    # 使用动态规划进行序列对齐
    z0_ids, z1_ids = _align_sequences(base_token_ids, target_token_ids, gap_token_id)

    return z0_ids, z1_ids, x0_ids, x1_ids


def _align_sequences(
    seq_0: List[int],
    seq_1: List[int],
    gap_token_id: int,
) -> Tuple[List[int], List[int]]:
    """使用动态规划对齐两个序列（最小编辑距离）.

    Args:
        seq_0: 基础序列
        seq_1: 目标序列
        gap_token_id: gap token 的 id

    Returns:
        aligned_0: 对齐后的 seq_0（含 gap token）
        aligned_1: 对齐后的 seq_1（含 gap token）
    """
    m, n = len(seq_0), len(seq_1)

    # DP 表初始化
    dp = [[i + j if i == 0 or j == 0 else 0 for j in range(n + 1)] for i in range(m + 1)]

    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_0[i - 1] == seq_1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # 回溯构建对齐序列
    aligned_0, aligned_1 = [], []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_0[i - 1] == seq_1[j - 1]:
            aligned_0.append(seq_0[i - 1])
            aligned_1.append(seq_1[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # 替换
            aligned_0.append(seq_0[i - 1])
            aligned_1.append(seq_1[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # 删除（在 seq_1 中插入 gap）
            aligned_0.append(seq_0[i - 1])
            aligned_1.append(gap_token_id)
            i -= 1
        else:
            # 插入（在 seq_0 中插入 gap）
            aligned_0.append(gap_token_id)
            aligned_1.append(seq_1[j - 1])
            j -= 1

    return aligned_0[::-1], aligned_1[::-1]
