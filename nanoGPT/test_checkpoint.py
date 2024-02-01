#!/usr/bin/env python3

"""
Test contents of checkpoint
"""

import os
import pickle
from contextlib import nullcontext

import tiktoken
import torch

from sub.char_tokenizer import CharacterTokenizer
from sub.config import COMPILE, DEVICE, DTYPE, INIT_FROM, TOP_K, VERB
from sub.model import GPT, GPTConfig
from sub.model_dist import split_parameters

script_dir = os.path.dirname(__file__)

dataset = "shakespeare"
dataset_name = os.path.splitext(dataset)[0]
data_dir = os.path.join(script_dir, "data", dataset_name)
out_dir = os.path.join(data_dir, "out")

if "cuda" in DEVICE:
    device_type = "cuda"
elif "mps" in DEVICE:
    device_type = "mps"
else:
    device_type = "cpu"

if __name__ == "__main__":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    print("Checkpoint breakdown:\n", checkpoint.keys())
    # KEYS: ['model', 'optimizer', 'model_args', 'iter_num', 'best_val_loss', 'config']
    print("")
    # print("Printing info about `checkpoint['config']`")
    # print("> Elements: ", checkpoint["config"])
    # print("")
    # print("Printing info about `checkpoint['model_args']`")
    # print("> Elements: ", checkpoint["model_args"])
    # print("")
    print("Printing info about `checkpoint['model']`")
    print("> Length: ", len(checkpoint["model"]))
    print("> Type: ", type(checkpoint["model"]))
    mod_keys = list(checkpoint["model"].keys())
    fname = os.path.join(script_dir, "tmp", "model_keys.txt")
    print("> Keys: ")
    with open(fname, "w") as f:
        for k in mod_keys:
            print(k)
            f.write(str(k) + "\n")
    keys_begin = [k.split(".")[0] for k in mod_keys]
    begin_once = list(set(keys_begin))
    # print("> Beginnings of keys: ", begin_once)

    # Print first element (first key)
    # print(f"Key: {mod_keys[0]} --> {checkpoint['model'][mod_keys[0]]}")

    print(split_parameters(checkpoint["model"], 2))
