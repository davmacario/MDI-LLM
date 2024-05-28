#!/usr/bin/env python3


from argparse import ArgumentParser
import os
import pickle
from pathlib import Path

import numpy as np
import requests
import tiktoken
import torch

from sub import Tokenizer
from sub.utils import load_dataset, split_dataset


def main(args):
    tok_path = Path(args.tokenizer)
    if not tok_path.is_dir():
        tok_path = tok_path.parent

    tokenizer = Tokenizer(tok_path)
    data_dir = Path(args.data)

    if not data_dir.is_dir():
        raise NotADirectoryError("Unable to find dataset dir")
    
    # Dataset file is the only ".txt" file inside
    data_file = None
    for f in data_dir.iterdir():
        if f.name.endswith((".txt", ".md")):
            data_file = f
            break

    assert data_file, f"Text file not found in {data_dir}!"

    data = load_dataset(data_file, tokenizer)  # Already a tensor!
    train_ids, val_ids = split_dataset(data, 0.9)
    vocab_size = tokenizer.vocab_size

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training data set has {len(train_ids):,} tokens")
    print(f"Validation data set has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(data_dir / "train.bin")
    val_ids.tofile(data_dir / "val.bin")

    print(f"Created files 'train.bin' and 'val.bin' in {args.data}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "data",
        metavar="DATA",
        type=str,
        help="""path of the data set. It should be a directory containing a text file
        from which to extract the training samples"""
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="./checkpoints/custom/NanoLlama",
        help="path to the directory containing the tokenizer configuration."
    )
    args = parser.parse_args()
    main(args)
