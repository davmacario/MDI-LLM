#!/usr/bin/env python3

import argparse
import os

import torch

from sub.config import DEVICE

script_dir = os.path.dirname(__file__)

"""
This script is used to perform "ckpt surgery", i.e., open existing model
checkpoints, inspect the parameters and possibly modify them (in case something
went wrong, to prevent re-running training).

Usage:
    python3 model_surgery.py --ckpt <path-to-ckpt>

Modifications:
    - If the dataset global parameter does not correspond to the parent folder,
        update it accordingly
"""

if __name__ == "__main__":
    ckpt_path_def = os.path.join(
        script_dir, "data", "shakespeare_bpe", "out", "ckpt_12layers_256ctx.pt"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=ckpt_path_def, help="Checkpoint path"
    )

    args = parser.parse_args()

    print(f"Loading ckpt from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=DEVICE)

    print("> ckpt.config.DATASET: ", ckpt["config"]["DATASET"])
    dataset_folder_name = os.path.basename(
        os.path.dirname(os.path.dirname(args.ckpt))
    )
    print("> Theoretical dataset name: ", dataset_folder_name)

    if ckpt["config"]["DATASET"] != dataset_folder_name:
        ckpt["config"]["DATASET"] = dataset_folder_name
        torch.save(ckpt, args.ckpt)
        print("Ckpt was updated (fixed dataset name)!")
