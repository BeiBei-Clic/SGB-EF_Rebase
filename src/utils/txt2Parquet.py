"""Convert JSON-line txt files to Parquet format for efficient loading."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def txt_to_parquet(txt_path: str, parquet_path: str, chunk_size: int = 10000) -> None:
    """Convert txt file to parquet format."""
    print(f"Reading {txt_path}...")
    data = [json.loads(line.strip()) for line in open(txt_path, 'r')]
    print(f"Loaded {len(data)} samples")

    print(f"Creating DataFrame...")
    df = pd.DataFrame(data)

    print(f"Writing to {parquet_path}...")
    df.to_parquet(parquet_path, index=False, compression='snappy')
    print(f"Done! Wrote {len(df)} rows")


def main():
    parser = argparse.ArgumentParser(description='Convert txt files to parquet format')
    parser.add_argument('input', help='Input txt file path')
    parser.add_argument('output', help='Output parquet file path')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for processing')
    args = parser.parse_args()

    txt_to_parquet(args.input, args.output, args.chunk_size)


if __name__ == '__main__':
    main()
