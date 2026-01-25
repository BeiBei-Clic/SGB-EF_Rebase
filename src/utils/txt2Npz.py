"""Convert JSON-line txt files to NPZ format."""

import argparse
import json

import numpy as np


def remove_gap_tokens(z_ids: list, gap_token: int = 0, pad_token: int = 4) -> list:
    """从对齐序列中移除 GAP token，得到未对齐序列.

    保留 BOS token 和有效 token，去掉 GAP token 和末尾的 PAD token.
    """
    result = []
    for token in z_ids:
        if token == pad_token:
            break  # 遇到 PAD 就停止（后面全是 padding）
        if token != gap_token:
            result.append(token)  # 保留 BOS 和有效 token，跳过 GAP
    return result


def txt_to_npz(txt_path: str, npz_path: str, compress: bool = True) -> None:
    data_list = [json.loads(line.strip()) for line in open(txt_path, 'r')]

    num_samples = len(data_list)
    input_dimensions = np.array([d['input_dimension'] for d in data_list], dtype=np.int64)
    x_values = np.array([d['x_values'] for d in data_list], dtype=np.float32)
    y_target = np.array([d['y_target'] for d in data_list], dtype=np.float32)

    z0_token_ids = np.empty(num_samples, dtype=object)
    z0_token_ids[:] = [np.array(d['z0_token_ids'], dtype=np.int64) for d in data_list]

    z1_token_ids = np.empty(num_samples, dtype=object)
    z1_token_ids[:] = [np.array(d['z1_token_ids'], dtype=np.int64) for d in data_list]

    x0_token_ids = np.empty(num_samples, dtype=object)
    x0_token_ids[:] = [np.array(remove_gap_tokens(d['z0_token_ids']), dtype=np.int64) for d in data_list]

    x1_token_ids = np.empty(num_samples, dtype=object)
    x1_token_ids[:] = [np.array(remove_gap_tokens(d['z1_token_ids']), dtype=np.int64) for d in data_list]

    if compress:
        np.savez_compressed(npz_path, input_dimensions=input_dimensions,
                            x_values=x_values, y_target=y_target,
                            z0_token_ids=z0_token_ids, z1_token_ids=z1_token_ids,
                            x0_token_ids=x0_token_ids, x1_token_ids=x1_token_ids)
    else:
        np.savez(npz_path, input_dimensions=input_dimensions,
                 x_values=x_values, y_target=y_target,
                 z0_token_ids=z0_token_ids, z1_token_ids=z1_token_ids,
                 x0_token_ids=x0_token_ids, x1_token_ids=x1_token_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input txt file')
    parser.add_argument('output', help='Output NPZ file')
    parser.add_argument('--no-compress', action='store_true')
    args = parser.parse_args()
    txt_to_npz(args.input, args.output, compress=not args.no_compress)


if __name__ == '__main__':
    main()
