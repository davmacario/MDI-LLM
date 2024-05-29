#!/usr/bin/env python3

"""
Training script - LitGPT model

Supports DistributedDataParallel through `torchrun --nproc-per-node <n> ./train.py`
"""

import gc
import inspect
import json
import math
import os
import pickle
import shutil
import time
import warnings
from argparse import ArgumentParser
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from sub import GPT
from sub.config import DEVICE, TrainingConfig
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

    # Distributed Data Parallel (data parallelism over multiple GPUs)
    ddp = False
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

    # ---
    # Extract args
    ckpt_dir = Path(args.ckpt)
    model_config_file = ckpt_dir / "model_config.yaml"
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
    train.compile = args.compile
    device = args.device
    # TODO: dtype

    # Data parallel training setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # DDP iff rank > -1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert train.gradient_accumulation_steps % ddp_world_size == 0
        train.gradient_accumulation_steps //= ddp_world_size
    else:
        # Cannot override device if using ddp
        device = args.device

    torch_device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    if "cuda" in device:
        device_type = "cuda"
    elif "mps" in device:
        device_type = "mps"
    else:
        device_type = "cpu"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type in {"cpu", "mps"}
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # ---------------------------------------------------------------------------------

    # Data loader (the partition needs to be created with prepare_data.py)
    train_data = np.memmap(dataset_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(dataset_dir / "val.bin", dtype=np.uint16, mode="r")

    # Init these up here, can override if INIT_FROM='resume'
    iter_num = 0
    best_val_loss = 1e9

    # Initialization of the model
    if init_from == "scratch":
        if not ckpt_dir:
            raise ValueError("Missing model checkpoint folder")

        # NOTE: we assume the training data has already been prepared and is stored
        # as .bin files -> TOKENIZER is NOT loaded here!
        if (
            not (ckpt_dir / "tokenizer.model").exists()
            and not (ckpt_dir / "tokenizer.json").exists()
        ):
            # Copy the tokenizer to the checkpoint directory to package
            if (dataset_dir / "tokenizer.model").exists():
                shutil.copy(dataset_dir / "tokenizer.model", ckpt_dir)
            elif (dataset_dir / "tokenizer.json").exists():
                shutil.copy(dataset_dir / "tokenizer.model", ckpt_dir)

        config, _ = load_from_pt(ckpt_dir, config_only=True)

        model = GPT(config)
        initialize_weights(model, n_layer=config.n_layer, n_embd=config.n_embd)
    elif init_from == "resume":
        # Look for checkpoint file
        if not ckpt_file.exists() or not ckpt_model.exists():
            raise FileNotFoundError("Unable to find training checkpoint!")
        config, wt = load_from_pt(ckpt_dir)
        assert wt, "Unable to load model parameters!"
        if args.verb:
            print(f"Resuming training from {ckpt_model}")
        model = GPT(config)
        model.load_state_dict(wt)

        state = torch.load(ckpt_file)
        # assert state["config"] == config
        if args.force_old:
            train = state["train_settings"]
        iter_num = state["iter_num"]
        if iter_num > train.max_iters:
            raise ValueError(
                f"Iteration number of pretrained model ({iter_num}) is greater than the maximum number of iterations specified ({train.max_iters})!"
            )
        best_val_loss = state["best_val_loss"]
    elif init_from.lower() in {"hf", "huggingface"}:
        # In this case, the ckpt_dir is the model name on huggingface hub
        ckpt_dir = script_dir / "checkpoints" / args.ckpt
        if master_process:
            os.makedirs(ckpt_dir, exist_ok=True)
        model_config_file = ckpt_dir / "model_config.yaml"
        ckpt_file = ckpt_dir / "train_ckpt.pkl"
        ckpt_model = ckpt_dir / "lit_model.pth"
        if ckpt_model.exists() and model_config_file.exists():
            print("Model was already downloaded!")
            config, wt = load_from_pt(ckpt_dir)
        else:
            config, wt = load_from_hf(
                repo_id=args.ckpt,
                access_token=args.hf_token if args.hf_token else os.getenv("HF_TOKEN"),
                dtype=dtype,
                checkpoint_dir=script_dir / "checkpoints",
            )
        assert wt, "Unable to load model parameters!"
        model = GPT(config)
        model.load_state_dict(wt)
    else:
        raise ValueError(f"Init type not supported: {args.init}")

    n_params = sum(p.numel() for p in model.parameters())
    if args.verb:
        print(f"Number of model parameters: {n_params:,}")

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
        print(f"> The model will {"not" if train.compile else ""} be compiled")
        print("")
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight

    if train.compile:
        if args.verb:
            print("Compiling model - this may take a while", end="\r")
        try:
            model = torch.compile(model)
            if args.verb:
                print("Model compiled!")
        except RuntimeError as e:
            warnings.warn(f"Unable to compile model! {e}")
    elif args.compile and not hasattr(torch, "compile"):
        from importlib.metadata import version

        warnings.warn(
            f"Installed torch version ({version('torch')}) does not support compiling models"
        )

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.learning_rate,
        weight_decay=train.weight_decay,
        betas=(train.beta1, train.beta2),
        fused=(fused_available and device_type == "cuda"),
    )
    if args.init == "resume":
        optimizer.load_state_dict(state["optimizer"])

    X, Y = get_batch(train_data, train.batch_size, args.device, config)
    t_start = time.time()
    local_iter = 0
    state = {}
    count_loss_incr = 0
    running_mfu = -1.0
    raw_model = model.module if ddp else model
    while iter_num <= train.max_iters:
        if VERB:
            print(f"Training iter {iter_num}")

        lr = get_lr(iter_num) if train.decay_lr else train.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if not iter_num % train.ckpt_interval and master_process:
            losses = estimate_loss(
                raw_model, train_data, val_data, 1, args.device, ctx=ctx
            )
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
                    # NOTE: will overwrite old model if resuming
                    torch.save(raw_model.state_dict(), ckpt_model)
            else:
                count_loss_incr += 1
                if args.patience is not None and count_loss_incr >= args.patience:
                    print(
                        f"No performance increase in the last {args.patience} iters - stopping!"
                    )
                    break

        if iter_num == 0 and train.eval_only:
            break

        # Gradient Accumulation
        for micro_step in range(train.gradient_accumulation_steps):
            if ddp:  # Sync gradients @ last micro step for DDP
                model.require_backward_grad_sync = (
                    micro_step == train.gradient_accumulation_steps - 1
                )
            with ctx:
                logits = model(X)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1
                )
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
        if not iter_num % train.log_interval and master_process:
            lossf = loss.item() * train.gradient_accumulation_steps
            if local_iter >= 5:
                # Estimate MFU
                mfu = raw_model.estimate_mfu(
                    train.batch_size * train.gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"Iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}, "
                f"MFU = {running_mfu * 100:.2f}%"
            )

        iter_num += 1
        local_iter += 1

    if ddp:
        torch.distributed.destroy_process_group()
    print("Training finished!")


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
        "--ckpt",
        type=str,
        default="./checkpoints/custom/NanoLlama/",
        help="""
        path to the model folder (only if loading from pretrained or training from
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
        "-F",
        "--force-old",
        action="store_true",
        help="""if resuming training ('--init resume'), force the old training settings
        - NOTE: this may cause issues if resuming training on a different computer""",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="training batch size (default=10)"
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="number of training iterations (default=100)",
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
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="""Huggingface Hub token to access restricted/private workspaces;
        not required if the HF_TOKEN env variable is set.""",
    )
    parser.add_argument("--seed", type=int, default=10137, help="random seed")
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        # Cleanup
        print("Training stopped by user!")
