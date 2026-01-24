"""Convert JSON-line txt files to NPZ format."""

import argparse
import json

import numpy as np


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

    if compress:
        np.savez_compressed(npz_path, input_dimensions=input_dimensions,
                            x_values=x_values, y_target=y_target,
                            z0_token_ids=z0_token_ids, z1_token_ids=z1_token_ids)
    else:
        np.savez(npz_path, input_dimensions=input_dimensions,
                 x_values=x_values, y_target=y_target,
                 z0_token_ids=z0_token_ids, z1_token_ids=z1_token_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input txt file')
    parser.add_argument('output', help='Output NPZ file')
    parser.add_argument('--no-compress', action='store_true')
    args = parser.parse_args()
    txt_to_npz(args.input, args.output, compress=not args.no_compress)


if __name__ == '__main__':
    main()
