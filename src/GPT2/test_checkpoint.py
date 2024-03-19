#!/usr/bin/env python3

"""
Test contents of checkpoint
"""

import argparse
import io
import os
import pickle
import sys
from contextlib import nullcontext

import tiktoken
import torch

from sub.char_tokenizer import CharacterTokenizer
from sub.config import DEVICE
from sub.model import GPT, GPTConfig
from sub.model_dist import FinisherNode
from sub.utils import (deserialize_params, get_obj_size, serialize_params,
                       split_parameters)

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
    """
    Example usage:
        python3 test_checkpoint.py --ckpt data/shakespeare_bpe/out/ckpt_12layers_128ctx.pt --split
    """
    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=ckpt_path, help="Checkpoint path")
    parser.add_argument(
        "--split",
        default=False,
        action="store_true",
        help="Perform model split",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    ckpt_sz = get_obj_size(checkpoint)
    print("Checkpoint size (in RAM): ", ckpt_sz)
    print("")

    print("Checkpoint breakdown:\n", checkpoint.keys())
    # KEYS: ['model', 'optimizer', 'model_args', 'iter_num', 'best_val_loss', 'config']
    print("")
    print("Printing info about `checkpoint['config']`")
    print("> Elements: ")
    for k, v in checkpoint["config"].items():
        print(f"\t{k}: {v}")
    # print("")
    # print("Printing info about `checkpoint['model_args']`")
    # print("> Elements: ", checkpoint["model_args"])
    # print("")
    print("Printing info about `checkpoint['model']`")
    print("> Length: ", len(checkpoint["model"]))
    print("> Type: ", type(checkpoint["model"]))
    mod_keys = list(checkpoint["model"].keys())
    fname = os.path.join(script_dir, "tmp", "model_keys.txt")
    # print("> Keys: ")
    with open(fname, "w") as f:
        for k in mod_keys:
            # print(k)
            f.write(str(k) + "\n")
    keys_begin = [k.split(".")[0] for k in mod_keys]
    begin_once = list(set(keys_begin))

    # Count the number of detected transformer layers
    layer_keys = [k for k in mod_keys if k.startswith("transformer.h")]
    layers_unique = list(set([".".join(k.split(".")[:3]) for k in layer_keys]))
    if not args.split:
        print(f"Number of transformer layers found in the model: {len(layers_unique)}")

    # print(
    #     f"\nProblematic keys:\nlm_head.weight: {checkpoint['model']['lm_head.weight'].shape}\nlm_head.bias: {checkpoint['model']['lm_head.bias'].shape}\n"
    # )
    # print("> Beginnings of keys: ", begin_once)

    # Print first element (first key)
    # print(f"Key: {mod_keys[0]} --> {checkpoint['model'][mod_keys[0]]}")

    buf = io.BytesIO()
    torch.save(checkpoint["model"], buf)
    buf.seek(0)
    print(f"Total model size (torch load to buffer): {len(buf.read())} B")

    if args.split:
        print("")
        # buf = io.BytesIO()
        # pickle.dump(checkpoint["model"], buf)
        # buf.seek(0)
        # print("Total model size (pickle): ", len(buf.read()))
        # print(
        #     "Total model size (serialized): ",
        #     get_obj_size(serialize_params(checkpoint["model"])),
        # )
        print("-> Splitting model")
        par_split, layers_info = split_parameters(checkpoint["model"], 3)

        start_sz = get_obj_size(serialize_params(par_split["starter"]))
        interm_sz = get_obj_size(serialize_params(par_split["intermediate"][0]))
        finish_sz = get_obj_size(serialize_params(par_split["finisher"]))
        print("Starter model size: ", start_sz)
        print("Intermediate model size: ", interm_sz)
        print("Finisher model size: ", finish_sz)
        print("Sum: ", start_sz + interm_sz + finish_sz)
        print("Model: ", get_obj_size(serialize_params(checkpoint["model"])))
