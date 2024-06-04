#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import torch

from sub import Tokenizer

"""
Prepare the OpenWebText dataset to be used to train the model.
"""
curr_dir = Path(os.path.dirname(__file__))


def main(args):
    # Number of workers in load_dataset
    out_path = args.data_path

    # Load tokenizer
    tok_path = Path(args.TOKENIZER_PATH)
    if not args.TOKENIZER_PATH.is_dir():
        tok_path = args.TOKENIZER_PATH.parent
    tokenizer = Tokenizer(tok_path)

    dataset = load_dataset("openwebtext", num_proc=args.nproc)

    split_ds = dataset["train"].train_test_split(
        test_size=0.0005, seed=1234, shuffle=True
    )
    split_ds["val"] = split_ds.pop("test")

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = tokenizer.encode(example["text"], device=args.device, bos=True, eos=True)
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {"ids": ids, "len": len(ids)}
        return out

    # Tokenize the dataset
    tokenized = split_ds.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=args.nproc,
    )

    # Concatenate all the ids in each dataset into one large file for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = args.data_path / f"{split}.bin"
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename.name}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "TOKENIZER_PATH",
        type=Path,
        help="path to the directory containing the tokenizer",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=curr_dir / "./data/openwebtext/",
        help="""directory where the data set will be stored; default:
        './data/openwebtext'""",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=None,
        help="torch device used to load tensors"
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=5,
        help="number of processes for tokenization (default: 5)"
    )
    args = parser.parse_args()

    main(args)
