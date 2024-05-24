#!/usr/bin/env python3

"""
Idea: define the model configuration in a yaml file as LitGPT does, then, depending on
the --init argument (scratch, resume), either create a model from scratch or resume from
a pretrained one (fine tuning).

The data set is handled in the same way as GPT-2, through train.bin and test.bin files.

Add possibility to init from hf, which downloads model and fine tunes it

MISSING:
- DDP
- Fine-tuning from compatible HF model
- Resuming training
"""

import gc
import inspect
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
from sub.config import DEVICE, DTYPE, TrainingConfig
from sub.model import CausalSelfAttention, LLaMAMLP
from sub.utils import (estimate_loss, get_batch, get_lr, load_from_hf,
                       load_from_pt)


def initialize_weights(model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(
                init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd)
            )

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(
                init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer)
            )


def main(args):
    script_dir = Path(os.path.dirname(__file__))
    # Extract args
    model_config_file = (
        Path(args.model_config) if args.model_config else None
    )  # None if HF init
    ckpt_dir = model_config_file.parent  # Where the model will be stored (FIXME)
    ckpt_file = ckpt_dir / "train_ckpt.pkl"
    ckpt_model = ckpt_dir / "lit_model.pth"
    dataset_dir = Path(args.dataset)
    dataset_name = dataset_dir.name
    init_from = args.init  # scratch / resume / hf
    VERB = args.verb
    train = TrainingConfig()
    train.batch_size = args.batch_size
    train.max_iters = args.max_iters
    train.ckpt_interval = args.ckpt_interval
    train.log_interval = args.log_interval
    train.gradient_accumulation_steps = args.grad_acc_steps
    train.device = args.device
    torch_device = torch.device(args.device)
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

    # TODO: store config parameters -- store train as dict in checkpoints (see later)
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

        if (ckpt_dir / "tokenizer.model").exists() or (
            ckpt_dir / "tokenizer.json"
        ).exists():
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
        n_params = sum(p.numel() for p in model.parameters())
        if args.verb:
            print(f"Number of model parameters: {n_params}")
    else:
        # TODO
        raise ValueError("Not implemented")

    model.to(torch_device)

    tokens_per_iter = (
        train.gradient_accumulation_steps * train.batch_size * model.max_seq_length
    )
    if VERB:
        print("Training configuration:")
        print("> Device: ", args.device)
        print("> Batch size: ", train.batch_size)
        print("> Gradient Accumulation steps: ", train.gradient_accumulation_steps)
        print("> Max epochs: ", train.max_iters)
        print("> Checkpoint update interval: ", train.ckpt_interval)
        print("")
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight

    if args.compile:
        model = torch.compile(model)

    if args.init != "resume":
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == "float16"))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train.learning_rate,
            weight_decay=train.weight_decay,
            betas=(train.beta1, train.beta2),
            fused=(fused_available and device_type == "cuda"),
        )
    else:
        raise ValueError("Not Implemented!")

    X, Y = get_batch(train_data, train.batch_size, args.device, config)
    t_start = time.time()
    local_iter = 0
    iter_num = 0  # FIXME: can be inferred from checkpoint if resuming
    state = {}  # TODO: store here info
    count_loss_incr = 0
    while iter_num <= train.max_iters:
        if VERB:
            print(f"Training iter {local_iter}")

        lr = get_lr(iter_num) if train.decay_lr else train.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if not iter_num % train.ckpt_interval:
            losses = estimate_loss(model, train_data, val_data, 1, args.device, ctx=ctx)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if losses["val"] <= best_val_loss or args.always_update:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    """
                    Checkpoint contents:
                    - optimizer: optimizer state dict
                    - train_settings: training configuration (see TrainingConfig class)
                    - iter_num
                    - best_val_loss
                    - config: full Config object of the model
                    + model stored as lit_model.pth

                    This should be the whole information necessary to resume training.
                    """
                    state = {
                        "optimizer": optimizer.state_dict(),
                        "train_settings": train,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"Saving state to {ckpt_file} and model to {ckpt_model}")
                    torch.save(state, ckpt_file)
                    torch.save(model.state_dict(), ckpt_model)
            else:
                count_loss_incr += 1
                if args.patience is not None and count_loss_incr > args.patience:
                    print(
                        f"No performance increase in the last {args.patience} iters - stopping!"
                    )
                    break

        if iter_num == 0 and train.eval_only:
            break

        # Gradient Accumulation
        for micro_step in range(train.gradient_accumulation_steps):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / train.gradient_accumulation_steps
            X, Y = get_batch(train_data, train.batch_size, args.device, config)
            # Backward pass + scaling if fp16 is used
            scaler.scale(loss).backward()

        # Gradient clipping
        if train.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train.grad_clip)

        # Step optimizer & scale
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # Flush gradients to free memory

        t1 = time.time()
        dt = t1 - t_start
        t_start = t1

        # Logging
        if not iter_num % train.log_interval:
            lossf = loss.item() * train.gradient_accumulation_steps
            if local_iter >= 5:
                # Estimate MFU
                pass
            print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}")

        iter_num += 1
        local_iter += 1

    print("Training stopped!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--verb", action="store_true", help="enable verbose")
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="if set, compile the model (Torch >= 2.0.0 required)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="./checkpoints/custom/NanoLlama/model_config.yaml",
        help="""
        path to the model config file (only if loading from pretrained or training from
        scratch
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/shakespeare",
        help="""training data set directory; it should contain the files 'train.bin' and
        'val.bin'""",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="scratch",
        help="""initialization - can be: 'scratch' (default) to initialize a new model
        from scratch given the model_config.yaml file, 'resume', to resume training from
        an existing checkpoint, or 'huggingface' to finetune a model from huggingface""",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="training batch size (default=10)"
    )
    parser.add_argument(
        "--max-iters", type=int, default=100, help="number of training iterations (default=100)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="""if set, it is the max number of checkpoint intervals without decreasing
        loss before the training loop is interrupted""",
    )
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=20,
        help="number of iterations between each checkpoint (default=20)",
    )
    parser.add_argument(
        "-au",
        "--always-update",
        action="store_true",
        help="""if set, always update the checkpoint, even if the validation accuracy
        decreases""",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="number of iterations between each log (default=10)",
    )
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=10,
        help="number of gradient accumulation steps (default=10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="device where to load the model for training",
    )
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        # Cleanup
        print("Training stopped by user!")
