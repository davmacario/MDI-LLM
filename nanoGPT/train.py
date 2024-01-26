#!/usr/bin/env python3

"""
Aadapted/rewritten from karpathy/nanoGPT/train.py
"""

import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from sub.config import (ALWAYS_SAVE_CHECKPOINT, BATCH_SIZE, BETA1, BETA2, BIAS,
                        BLOCK_SIZE, COMPILE, DECAY_LR, DEVICE, DROPOUT, DTYPE,
                        EVAL_INTERVAL, EVAL_ONLY, GRAD_CLIP,
                        GRADIENT_ACCUMULATION_STEPS, INIT_FROM, LEARNING_RATE,
                        LOG_INTERVAL, MAX_ITERS, N_EMBD, N_HEADS, N_LAYER,
                        VERB, WEIGHT_DECAY)
from sub.data_loader import get_batch
from sub.model import GPT, GPTConfig
from sub.utils import estimate_loss, get_lr

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O configuration
script_dir = os.path.dirname(__file__)
out_dir = os.path.join(script_dir, "out")  # Checkpoints

# TODO: decide what to do with wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())

# DATA
# dataset = "openwebtext"
dataset = "shakespeare"
# dataset = "divina_commedia"

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# exec(open("configurator.py").read())  # overrides from cmd or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
master_process = True
seed_offset = 0
ddp_world_size = 1

tokens_per_iter = (
    GRADIENT_ACCUMULATION_STEPS * ddp_world_size * BATCH_SIZE * BLOCK_SIZE
)
if VERB:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# for later use in torch.autocast:
if "cuda" in DEVICE:
    device_type = "cuda"
elif "mps" in DEVICE:
    device_type = "mps"
else:
    device_type = "cpu"
# note: float16 data type will automatically use a GradScaler
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

# Poor man's data loader
dataset_name = os.path.splitext(dataset)[0]
data_dir = os.path.join(script_dir, "data", dataset_name)
out_dir = os.path.join(data_dir, "out")
if master_process:
    os.makedirs(out_dir, exist_ok=True)
train_data = np.memmap(
    os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
)
val_data = np.memmap(
    os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
)

# Init these up here, can override if INIT_FROM='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# ----------------------------------------------------

# Iff char-based tokenizer was used, look for metadata
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model init
model_args = dict(
    n_layer=N_LAYER,
    n_head=N_HEADS,
    n_embd=N_EMBD,
    block_size=BLOCK_SIZE,
    bias=BIAS,
    vocab_size=None,
    dropout=DROPOUT,
)

if INIT_FROM == "scratch":
    # Init a new model from scratch
    print("Initializing a new model from scratch")

    # Determine the vocab size
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = (
        meta_vocab_size if meta_vocab_size is not None else 50304
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif INIT_FROM == "resume":
    # Resume training from a checkpoint (fine-tune).
    print(f"Resuming training from {out_dir}")

    # TODO: review position of .pt
    # ckpt_path = os.path.join(out_dir, "ckpt.pt")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    checkpoint_model_args = checkpoint["model_args"]
    if meta_vocab_size is not None:
        assert (
            checkpoint_model_args["vocab_size"] == meta_vocab_size
        ), "The vocab sizes do not match!"
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
        "vocab_size",
    ]:
        model_args[k] = checkpoint_model_args[k]
    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # ---
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    # ---
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"Invalid initialization: {INIT_FROM}")

# Crop down the model block size if desired, using model surgery
if BLOCK_SIZE < model.config.block_size:
    if VERB:
        print(
            f"Cropping down block size from {model.config.block_size} to {BLOCK_SIZE}"
        )
    model.crop_block_size(BLOCK_SIZE)
    model.config.block_size = BLOCK_SIZE

# Should not be needed here - model is moved to device at initialization
# model.to(device)

# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == "float16"))

# Optimizer
optimizer = model.configure_optimizers(
    WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), device_type
)
if INIT_FROM == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if COMPILE:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# Logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ------- Training loop ---------------------------

X, Y = get_batch(train_data, gptconf)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
# FIX:
# raw_model = model.module if ddp else model  # unwrap DDP container if needed
raw_model = model
running_mfu = -1.0
while iter_num <= MAX_ITERS:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if DECAY_LR else LEARNING_RATE
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % EVAL_INTERVAL == 0 and master_process:
        losses = estimate_loss(model, train_data, val_data)
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if losses["val"] < best_val_loss or ALWAYS_SAVE_CHECKPOINT:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(
                    f"saving checkpoint to {os.path.join(out_dir, 'ckpt.pt')}"
                )
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and EVAL_ONLY:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / GRADIENT_ACCUMULATION_STEPS
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch(train_data, gptconf)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if GRAD_CLIP != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % LOG_INTERVAL == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * GRADIENT_ACCUMULATION_STEPS
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(
                BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, dt
            )
            running_mfu = (
                mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            )
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1
