#!/usr/bin/env python3

"""
Aadapted/rewritten from karpathy/nanoGPT/train.py
"""

import gc
import json
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from sub.config import (ALWAYS_SAVE_CHECKPOINT, BETA1, BETA2, BIAS, BLOCK_SIZE,
                        COMPILE, DECAY_LR, DEVICE, DROPOUT, DTYPE, EVAL_ONLY,
                        GRAD_CLIP, LEARNING_RATE, N_EMBD, N_HEADS, N_LAYER,
                        WEIGHT_DECAY)
from sub.data_loader import get_batch
from sub.model import GPT, GPTConfig
from sub.parser import parse_args
from sub.utils import estimate_loss, get_lr

# -----------------------------------------------------------------------------
# I/O configuration
script_dir = os.path.dirname(__file__)

DATASET = "shakespeare"  # NOTE: dataset *name*


def main() -> int:
    global DATASET
    # various inits, derived attributes, I/O setup
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

    # OVERRIDE globals with arguments
    args = parse_args()

    BATCH_SIZE = args.batch_size
    INIT_FROM = args.init
    MAX_ITERS = args.max_iters
    LOG_INTERVAL = args.log_interval
    VERB = args.verb
    CKPT_INTERVAL = args.ckpt_interval

    # --dataset: path of the dataset (folder where to find train.bin & test.bin)
    if args.dataset is not None:
        assert os.path.isdir(args.dataset) and os.path.exists(args.dataset)
        dataset_dir = args.dataset
        DATASET = os.path.basename(args.dataset)
        out_dir = os.path.join(dataset_dir, "out")
    else:
        # Default dataset value
        dataset_dir = os.path.join(script_dir, "data", DATASET)

    # --ckpt: checkpoint file (either output or from which to resume)
    # NOTE: can be in different dir than the dataset's
    if args.ckpt is not None:
        out_dir = os.path.dirname(args.ckpt)
        ckpt_path = args.ckpt
        if args.dataset is None:
            # Update dataset directory to the parent of out/ckpt[...].pt - default
            out_dir = os.path.dirname(ckpt_path)
            dataset_dir = os.path.dirname(out_dir)
            DATASET = os.path.basename(dataset_dir)
    else:
        out_dir = os.path.join(dataset_dir, "out")
        ckpt_path = os.path.join(out_dir, "ckpt.pt")

    GRADIENT_ACCUMULATION_STEPS = args.grad_acc_steps

    # Setting up paths
    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    # Store global configuration parameters (all of the above)
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]

    # Store globals in the config of the checkpoint
    config = {k: globals()[k] for k in config_keys}  # useful for logging

    # Data parallelism settings
    ddp = ("cuda" in DEVICE) and (torch.cuda.device_count() > 1)
    # ddp2 = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    # assert ddp == ddp2, "Not the same DDP value"
    if ddp:
        torch.distributed.init_process_group(backend="nccl")

    # -------------------------------------------------------------------------

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

    # Data loader (the partition needs to be created with prepare_data.py)
    train_data = np.memmap(
        os.path.join(dataset_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(dataset_dir, "val.bin"), dtype=np.uint16, mode="r"
    )

    # Init these up here, can override if INIT_FROM='resume'
    iter_num = 0
    best_val_loss = 1e9

    # ----------------------------------------------------

    # Iff char-based tokenizer was used, look for metadata
    meta_path = os.path.join(dataset_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # Look for bpe tokenizer
    vocab_path = os.path.join(dataset_dir, "encoder.json")
    merges_path = os.path.join(dataset_dir, "merges.bpe")
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        with open(vocab_path, "r") as f:
            tok_vocab = json.load(f)
            f.close()
        meta_vocab_size = len(tok_vocab)
        print(f"Found vocab_size = {meta_vocab_size} (inside {vocab_path})")

    # Model init
    model_args = dict(
        block_size=BLOCK_SIZE,
        vocab_size=None,
        n_layer=N_LAYER,
        n_head=N_HEADS,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
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

        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        checkpoint_model_args = checkpoint["model_args"]
        if meta_vocab_size is not None:
            assert (
                checkpoint_model_args["vocab_size"] == meta_vocab_size
            ), "The vocab sizes do not match!"
        # force these config attributes to be equal otherwise we can't even
        # resume training the rest of the attributes (e.g. dropout) can stay as
        # desired from command line
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
    elif INIT_FROM.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {INIT_FROM}")
        # Initialize from OpenAI GPT-2 weights (Huggingface)
        override_args = dict(dropout=DROPOUT)
        model = GPT.from_pretrained(INIT_FROM, override_args)
        # Read params to be stored in ckpt
        for k in [
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "bias",
            "vocab_size",
        ]:
            model_args[k] = getattr(model.config, k)
        gptconf = model.config
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

    # Move model to device (HERE! Not in __init__)
    model = model.to(DEVICE)
    # Print model settings
    if VERB:
        print("Model settings:")
        for k, v in model_args.items():
            print(f"> {k}: {v}")

    # Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == "float16"))

    # Optimizer
    optimizer = model.configure_optimizers(
        WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), device_type
    )
    if INIT_FROM == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])

        try:
            # Free up memory
            del state_dict
            del checkpoint
        except:
            pass
        finally:
            state_dict = None
            checkpoint = None
            torch.cuda.empty_cache()
            gc.collect()

    # Distributed
    if ddp:
        model = DDP(model)

    # compile the model
    if COMPILE:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # ------- Training loop ---------------------------

    X, Y = get_batch(train_data, BATCH_SIZE, DEVICE, gptconf)
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    count_loss_incr = 0
    while iter_num <= MAX_ITERS:
        if VERB:
            print(f"> Training iter {local_iter_num}")
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if DECAY_LR else LEARNING_RATE
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % CKPT_INTERVAL == 0 and master_process:
            losses = estimate_loss(
                model, train_data, val_data, BATCH_SIZE, DEVICE, **{"ctx": ctx}
            )
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss or ALWAYS_SAVE_CHECKPOINT:
                count_loss_incr = 0
                # Only store ckpt if loss has decreased
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    # Prevent loss of old parameters - do not overwrite
                    if INIT_FROM != "scratch":
                        ckpt_path_upd = os.path.join(
                            out_dir,
                            f"{os.path.splitext(os.path.basename(ckpt_path))[0]}_upd.pt",
                        )
                        print(f"Saving checkpoint to {ckpt_path_upd}")
                        torch.save(checkpoint, ckpt_path_upd)
                    else:
                        print(f"Saving checkpoint to {ckpt_path}")
                        torch.save(checkpoint, ckpt_path)
            elif losses["val"] >= best_val_loss:
                count_loss_incr += 1
                # If the validation loss has been increasing, stop
                if count_loss_incr > 10:
                    break

        if iter_num == 0 and EVAL_ONLY:
            # Exit after 1 evaluation of the loss
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
            # Missing: ddp update step
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
                # scale the loss to account for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(train_data, BATCH_SIZE, DEVICE, gptconf)
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
                mfu = model.estimate_mfu(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, dt)
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

    print("Training stoped!")
    return 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training stopped!")
