#!/usr/bin/env python3

"""
Aadapted/rewritten from karpathy/nanoGPT/train.py

---

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from sub import CharacterTokenizer
from sub.config import (ALWAYS_SAVE_CHECKPOINT, BACKEND, BATCH_SIZE, BETA1,
                        BETA2, BLOCK_SIZE, COMPILE, DEBUG, DECAY_LR, DEVICE,
                        DTYPE, EVAL_INTERVAL, EVAL_ITERS, EVAL_ONLY, GRAD_CLIP,
                        GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE,
                        LOG_INTERVAL, MAX_ITERS, N_EMBD, N_HEADS, N_ITER_TRAIN,
                        N_LAYER, VERB, WEIGHT_DECAY)
from sub.data_loader import get_batch, load_dataset, split_dataset
from sub.model import GPT, GPTConfig
from sub.utils import estimate_loss, get_lr

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O configuration
script_dir = os.path.dirname(__file__)
out_dir = os.path.join(script_dir, "out")  # Checkpoints

init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

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
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    assert False, "Shouldn't be here!"
    init_process_group(backend=backend)
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
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = (
    GRADIENT_ACCUMULATION_STEPS * ddp_world_size * BATCH_SIZE * BLOCK_SIZE
)
if VERB:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
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
ctx = nullcontext()
#     nullcontext()
#     if device_type == "cpu"
#     else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# )

# poor man's data loader
dataset_name = os.path.splitext(dataset)[0]
data_dir = os.path.join(script_dir, "data", dataset_name)
train_data = np.memmap(
    os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
)
val_data = np.memmap(
    os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
)

# TODO: ensure compatibility of 'get_batch' definition
# def get_batch(split):
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
#     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
#     if device_type == 'cuda':
#         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
#         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#     else:
#         x, y = x.to(device), y.to(device)
#     return x, y

# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# Iff char-based tokenizer was used, look for metadata
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# ----------------------------------------------------

# Model init
# model_args = dict(
#     n_layer=n_layer,
#     n_head=n_head,
#     n_embd=n_embd,
#     block_size=block_size,
#     bias=bias,
#     vocab_size=None,
#     dropout=dropout,
# )
# Start with model_args from command line
gptconf = GPTConfig()
gptconf.vocab_size = None
model_args = dict(
    n_layer=gptconf.n_layer,
    n_head=gptconf.n_head,
    n_embd=gptconf.n_embd,
    block_size=gptconf.block_size,
    bias=gptconf.bias,
    vocab_size=gptconf.vocab_size,
    dropout=gptconf.dropout,
)

if init_from == "scratch":
    # Init a new model from scratch
    print("Initializing a new model from scratch")

    # Determine the vocab size
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    gptconf.vocab_size = (
        meta_vocab_size if meta_vocab_size is not None else 50304
    )
    model = GPT(gptconf)
elif init_from == "resume":
    # Resume training from a checkpoint (fine-tune).
    print(f"Resuming training from {out_dir}")

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    checkpoint_model_args = checkpoint["model_args"]
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
    gptconf = GPTConfig(**model_args)  # FIXME - not sure it is an issue tho
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
# elif init_from.startswith("gpt2"):
#     print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
#     # initialize from OpenAI GPT-2 weights
#     override_args = dict(dropout=dropout)
#     model = GPT.from_pretrained(init_from, override_args)
#     # read off the created config params, so we can store them into checkpoint correctly
#     for k in [
#         "n_layer",
#         "n_head",
#         "n_embd",
#         "block_size",
#         "bias",
#         "vocab_size",
#     ]:
#         model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Invalid initialization: {init_from}")

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
# TODO: add these parameters to the config file for sure
optimizer = model.configure_optimizers(
    WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if COMPILE:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# Wrap model into DDP container
if ddp:
    assert False, "Shouldn't be here!"
    model = DDP(model, device_ids=[ddp_local_rank])

# TODO: ensure compatibility of this function
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             with ctx:
#                 logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out

# logging
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
while True:
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
        # if wandb_log:
        #     wandb.log(
        #         {
        #             "iter": iter_num,
        #             "train/loss": losses["train"],
        #             "val/loss": losses["val"],
        #             "lr": lr,
        #             "mfu": running_mfu * 100,  # convert to percentage
        #         }
        #     )
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
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and EVAL_ONLY:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == GRADIENT_ACCUMULATION_STEPS - 1
            )
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

    # termination conditions
    if iter_num > MAX_ITERS:
        break

if ddp:
    destroy_process_group()
