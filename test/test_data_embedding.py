"""测试数据加载与嵌入流程。"""

import sys
import torch

sys.path.insert(0, '/home/xyh/Symbolic_Regression/Diffusion-Based/SGB-EF_Rebase')

from src.data_loader.data_loader import SRDataLoader
from src.model.data_embedding import SetEncoder, prepare_encoder_input


def test_data_loader():
    """1. 数据加载测试"""
    print("=== 1. 测试数据加载 ===")
    loader = SRDataLoader(
        npz_path='data/test_sample.npz',
        batch_size=2,
        shuffle=False,
    )

    # 获取第一个 batch
    batch = next(iter(loader))
    z_0, z_1, t, padding_mask, x_values, y_target = batch

    print(f"z_0 shape: {z_0.shape}")
    print(f"z_1 shape: {z_1.shape}")
    print(f"t shape: {t.shape}")
    print(f"padding_mask shape: {padding_mask.shape}")
    print(f"x_values shape: {x_values.shape}")  # (B, n_points, input_dim)
    print(f"y_target shape: {y_target.shape}")  # (B, n_points)

    assert x_values.shape[0] == 2, f"Expected batch_size=2, got {x_values.shape[0]}"
    assert x_values.shape[1] == 100, f"Expected n_points=100, got {x_values.shape[1]}"
    assert x_values.shape[2] == 3, f"Expected input_dim=3, got {x_values.shape[2]}"
    assert y_target.shape == (2, 100), f"Expected (2, 100), got {y_target.shape}"

    print("  数据加载测试通过\n")
    return batch


def test_prepare_encoder_input(batch):
    """2. 数据准备测试"""
    print("=== 2. 测试数据准备 ===")
    z_0, z_1, t, padding_mask, x_values, y_target = batch

    encoder_input = prepare_encoder_input(x_values, y_target)

    print(f"encoder_input shape: {encoder_input.shape}")

    assert encoder_input.shape == (2, 100, 4), f"Expected (2, 100, 4), got {encoder_input.shape}"

    print("  数据准备测试通过\n")
    return encoder_input


def test_set_encoder(encoder_input):
    """3. SetEncoder 嵌入测试"""
    print("=== 3. 测试 SetEncoder 嵌入 ===")
    encoder = SetEncoder(dim_input=4)  # 3 (x_values) + 1 (y_target)

    embedded_features = encoder(encoder_input)

    print(f"embedded_features shape: {embedded_features.shape}")

    # SetEncoder 输出: (B, num_features, dim_hidden)
    # 默认 num_features=1, dim_hidden=128
    assert embedded_features.shape[0] == 2, f"Expected batch_size=2, got {embedded_features.shape[0]}"
    assert embedded_features.shape[1] == 1, f"Expected num_features=1, got {embedded_features.shape[1]}"
    assert embedded_features.shape[2] == 128, f"Expected dim_hidden=128, got {embedded_features.shape[2]}"

    print("  SetEncoder 嵌入测试通过\n")


def main():
    """完整流程测试"""
    print("测试数据加载与嵌入流程\n")

    # 数据流: data/test_sample.npz -> SRDataLoader -> batch
    batch = test_data_loader()

    # 数据准备: prepare_encoder_input(x, y) -> encoder_input
    encoder_input = test_prepare_encoder_input(batch)

    # SetEncoder 嵌入: SetEncoder -> embedded_features
    test_set_encoder(encoder_input)

    print("=== 所有测试通过 ===")


if __name__ == '__main__':
    main()
