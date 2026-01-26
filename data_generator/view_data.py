"""查看生成数据集的内容."""

import argparse
import numpy as np
from src.model.vocab import Vocabulary


def view_dataset(path: str, num_samples: int = 3):
    """查看数据集内容.

    Args:
        path: NPZ 文件路径
        num_samples: 显示样本数量
    """
    data = np.load(path)
    num_variables = data['x_values'].shape[2]
    vocab = Vocabulary(num_variables=num_variables)

    print(f"=== 数据集: {path} ===")
    print(f"样本数: {data['input_dimensions'].shape[0]}")
    print(f"序列长度: {data['z0_token_ids'].shape[1]}")
    print(f"采样点数: {data['y_target'].shape[1]}")
    print(f"变量数: {num_variables}")
    print()

    # 统计有效 token 数
    for name in ['x0_token_ids', 'x1_token_ids', 'z0_token_ids', 'z1_token_ids']:
        arr = data[name]
        non_pad = np.sum(arr != vocab.pad_token, axis=1)
        print(f"{name:15} 平均长度: {non_pad.mean():.2f}, 最小: {non_pad.min()}, 最大: {non_pad.max()}")

    print()

    # 显示前 N 个样本
    print(f"=== 前 {num_samples} 个样本 ===")
    for i in range(min(num_samples, data['input_dimensions'].shape[0])):
        print(f"\n--- 样本 {i} ---")

        # 显示 token 序列
        for name in ['x0_token_ids', 'x1_token_ids', 'z0_token_ids', 'z1_token_ids']:
            ids = data[name][i]
            valid_ids = ids[ids != vocab.pad_token]
            tokens = [vocab.id_to_token(int(id)) for id in valid_ids]
            ids_str = ' '.join(str(int(id)) for id in valid_ids)
            tokens_str = ' '.join(tokens)
            print(f"{name:15}: {tokens_str}")
            print(f"{name:15} ids: {ids_str}")

        # 显示采样点和目标值（前 5 个点）
        print(f"x_values (前5点): {data['x_values'][i, :5].tolist()}")
        print(f"y_target (前5点): {data['y_target'][i, :5].tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看生成数据集")
    parser.add_argument("--path", type=str, default="data/train_data.npz", help="数据文件路径")
    parser.add_argument("--num-samples", type=int, default=3, help="显示样本数量")

    args = parser.parse_args()

    view_dataset(args.path, args.num_samples)
