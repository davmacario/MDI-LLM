#!/usr/bin/env python3

"""
Idea: define the model configuration in a yaml file as LitGPT does, then, depending on
the --init argument (scratch, resume), either create a model from scratch or resume from
a pretrained one (fine tuning).

The data set is handled in the same way as GPT-2, through train.bin and test.bin files.

Add possibility to init from hf, which downloads model and fine tunes it
"""

import gc
import json
import math
import os
import pickle
import shutil
import time
from argparse import ArgumentParser
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sub import GPT, Config, Tokenizer
from sub.model import LLaMAMLP, CausalSelfAttention
from sub.utils import get_batch, get_lr, estimate_loss, load_from_pt, load_from_hf
from sub.config import DTYPE, TrainingConfig


def initialize_weights(model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

def main(args):
    script_dir = Path(os.path.dirname(__file__))
    # Extract args
    model_config_file = Path(args.model_config) if args.model_config else None  # None if HF init
    ckpt_dir = model_config_file.parent  # Where the model will be stored (FIXME)
    BATCH_SIZE = args.batch_size
    init_from = args.init  # scratch / resume / hf
    MAX_ITERS = args.max_iters
    VERB = args.verb
    CKPT_INTERVAL = args.ckpt_interval
    dataset_dir = Path(args.dataset)
    dataset_name = dataset_dir.name
    GRADIENT_ACCUMULATION_STEPS = args.grad_acc_steps
    device = torch.device(args.device)
    train = TrainingConfig()
    # TODO: dtype

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    if "cuda" in args.device:
        device_type = "cuda"
    elif "mps" in args.device:
        device_type = "mps"
    else:
        device_type = "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[DTYPE]
    ctx = (
        nullcontext()
        if device_type in {"cpu", "mps"}
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # TODO: store config parameters
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    config = {k: globals()[k] for k in config_keys}  # useful for logging

    # ---------------------------------------------------------------------------------

    # Data loader (the partition needs to be created with prepare_data.py)
    train_data = np.memmap(dataset_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(dataset_dir / "val.bin", dtype=np.uint16, mode="r")

    # Init these up here, can override if INIT_FROM='resume'
    iter_num = 0
    best_val_loss = 1e9

    # TODO: load correct tokenizer - ideally, place tokenizer file in data folder or in
    # checkpoint folder (same as config)

    # Initialization of the model
    if init_from == "scratch":
        if not model_config_file:
            raise ValueError("Missing model config file")

        if (ckpt_dir / "tokenizer.model").exists() or (ckpt_dir / "tokenizer.json").exists():
            tokenizer = Tokenizer(ckpt_dir)
        else:
            tokenizer = Tokenizer(dataset_dir)
            # Copy the tokenizer to the checkpoint directory to package
            if (dataset_dir / "tokenizer.model").exists():
                shutil.copy(dataset_dir / "tokenizer.model", ckpt_dir)
            elif (dataset_dir / "tokenizer.json").exists():
                shutil.copy(dataset_dir / "tokenizer.model", ckpt_dir)
        
        config, _ = load_from_pt(ckpt_dir, config_only=True)

        model = GPT(config)
        initialize_weights(model, n_layer=config.n_layer, n_embd=config.n_embd)
    else:
        # TODO
        raise ValueError("Not implemented")

    model.to(device)

    tokens_per_iter = (
        GRADIENT_ACCUMULATION_STEPS *  BATCH_SIZE * model.max_seq_length
    )
    if VERB:
        print("Training configuration:")
        print("> Device: ", args.device)
        print("> Batch size: ", BATCH_SIZE)
        print("> Gradient Accumulation steps: ", GRADIENT_ACCUMULATION_STEPS)
        print("> Max epochs: ", MAX_ITERS)
        print("> Checkpoint update interval: ", CKPT_INTERVAL)
        print("")
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")


    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight

    if args.compile:
        model = torch.compile(model)

    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == "float16"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.learning_rate,
        weight_decay=train.weight_decay,
        betas=(train.beta1, train.beta2),
        fused=(device_type == "cuda"),
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        # Cleanup
        print("Training stopped!")
