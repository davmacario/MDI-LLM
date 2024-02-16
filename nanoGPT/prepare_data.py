#!/usr/bin/env python3

"""
This script is used to prepare a data set by loading it, tokenizing, encoding,
splitting it and storing it as .bin files in the correct position.

The .bin files are ready to be used by 'train.py'
"""

import os
import pickle

import numpy as np
import requests
import tiktoken
import torch

from sub import CharacterTokenizer
from sub.config import VERB
from sub.data_loader import load_dataset, split_dataset

CURR_DIR = os.path.dirname(__file__)


def main():
    # TODO: add arg parsing (select tokenizer and pass data dir)
    # data_set_dir = os.path.join(CURR_DIR, "data", "shakespeare")
    data_set_dir = os.path.join(CURR_DIR, "data", "shakespeare_bpe")
    # data_set_dir = os.path.join(CURR_DIR, "data", "divina_commedia")

    # Dataset file is the only ".txt" file inside
    data_file = None
    for f in os.listdir(data_set_dir):
        if f.endswith(".txt"):
            data_file = f
            in_file = os.path.join(data_set_dir, data_file)
            break

    if data_file is None:
        in_file = os.path.join(data_set_dir, "input.txt")
        if not os.path.exists(in_file):
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(in_file, "w") as f:
                f.write(requests.get(data_url).text)

    # tokenizer = CharacterTokenizer()
    tokenizer = tiktoken.get_encoding("gpt2")
    data_lst = load_dataset(in_file, tokenizer)  # data_lst is a list
    # Move to tensor here
    data = torch.tensor(data_lst, dtype=torch.long)
    vocab_size = tokenizer.n_vocab
    train_ids, val_ids = split_dataset(data, 0.9)
    if VERB:
        print(f"Vocabulary size - char tokenizer: {vocab_size}")
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

    # Export to .bin
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_set_dir, "train.bin"))
    val_ids.tofile(os.path.join(data_set_dir, "val.bin"))

    # Store tokenizer metadata if of type CharacterTokenizer (alphabet and
    # encodings change with data set)
    if type(tokenizer) == CharacterTokenizer:
        if VERB:
            print("Dumping character-based tokenizer metadata")
        meta = {
            "vocab_size": vocab_size,
            "itos": tokenizer.itos,
            "stoi": tokenizer.stoi,
        }
        with open(os.path.join(data_set_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)


if __name__ == "__main__":
    main()
