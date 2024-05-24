#!/usr/bin/env python3

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

"""
Prepare the OpenWebText dataset to be used to train the model.

TODO - ADAPT
"""

curr_dir = os.path.dirname(__file__)
# Number of workers in load_dataset
num_proc = 5
num_proc_load_ds = 2

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--out-dir",
    type=str,
    default=curr_dir,
    help="Output directory where to place the data set splits. Default: script directory",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        raise NotADirectoryError(f"{args.out_dir} is not a directory")

    dataset = load_dataset("openwebtext", num_proc=num_proc_load_ds)

    split_ds = dataset["train"].train_test_split(
        test_size=0.0005, seed=1234, shuffle=True
    )
    split_ds["val"] = split_ds.pop("test")

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(
            example["text"]
        )  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_ds.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(args.out_dir, f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
