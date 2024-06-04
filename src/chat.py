#!/usr/bin/env python3

"""
Rewrite of 'sample.py' enabling chat with LLM.
"""

import cProfile
import gc
import os
import pstats
import time
import warnings
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
from typing import Generator, Iterator

import torch

from sub import GPT, PromptStyle, Tokenizer
from sub.config import TEMPERATURE, TOP_K
from sub.prompts import get_user_prompt, has_prompt_style, load_prompt_style
from sub.utils import find_eot, load_from_pt, plot_tokens_per_time

script_dir = Path(os.path.dirname(__file__))


def interactive_prompt(prompt_style: PromptStyle) -> str:
    """
    Query the user interactively
    """
    user_input = input(">> User: ")
    return prompt_style.apply(user_input)


def decode(
    tokenizer: Tokenizer, token_stream: Iterator[torch.Tensor], device: torch.device
) -> int:
    tokens_generated = 0
    # Need to re-decode all text at each new token to insert spaces correctly
    so_far = torch.tensor([], dtype=torch.long, device=device)
    decoded_so_far = ""
    try:
        for token in token_stream:
            so_far = so_far.to(device=token.device)
            so_far = torch.cat((so_far, token.view(-1)))
            decoded_new = tokenizer.decode(so_far)
            print(decoded_new[len(decoded_so_far) :], end="", flush=True)
            decoded_so_far = decoded_new
            tokens_generated += 1
    except KeyboardInterrupt:
        # support stopping generation
        return tokens_generated
    return tokens_generated


def main(args):
    # Set up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    checkpoint_dir = Path(args.ckpt)
    model_type = checkpoint_dir.name
    checkpoint_path = args.ckpt / "lit_model.pth"  # Requires ckpt to be converted
    if not checkpoint_path.is_file() and (
        (checkpoint_dir / "model.bin").is_file()
        or (checkpoint_dir / "pytorch_model.bin.index.json").is_file()
    ):
        # Weights are there but in wrong format
        from sub.utils.convert_hf_checkpoint import convert_hf_checkpoint

        convert_hf_checkpoint(checkpoint_dir=checkpoint_dir, dtype=dtype)

    assert checkpoint_path.is_file(), "Something went wrong in weight conversion"

    # --------------------------------------------------------------------------
    # For later use in torch.autocast:
    if "cuda" in args.device:
        device_type = "cuda"
    elif "mps" in args.device:
        device_type = "mps"
    else:
        device_type = "cpu"
    if args.verb:
        print(f"Using {args.device}")
        print(f"Device type: {device_type}")
    ctx = (  # Use autocast if on cuda or cpu (MPS not supported yet)
        nullcontext()
        if device_type == "mps" or not dtype == "bfloat16"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )
    torch_device = torch.device(args.device)

    # Model setup
    config, wt = load_from_pt(checkpoint_dir)
    assert wt is not None
    model = GPT(config)

    model_dtype = torch.float32
    if all([v.dtype == torch.float16 for v in wt.values()]):
        model_dtype = torch.float16
    elif all([v.dtype == torch.bfloat16 for v in wt.values()]):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16

    if model_dtype in {torch.float16, torch.bfloat16}:
        model = model.to(model_dtype)
    model.load_state_dict(wt)
    del wt
    gc.collect()
    model.to(torch_device)

    n_params = sum(p.numel() for p in model.parameters())
    if args.verb:
        print(f"Number of model parameters: {n_params:,}")

    # NOTE: by increasing the batch size, the model can generate more samples together
    # but this would not be fair compared to MDI, as we could raise the batch size
    # there as well; instead, we generate individual samples multiple times

    # model.set_kv_cache(batch_size=batch_size)  # process samples together
    model.set_kv_cache(
        batch_size=1, device=torch_device
    )  # Re-set cache for every sample

    # Compile model + catch exception if unsupported (Python 3.12 currently)
    if args.compile and hasattr(torch, "compile"):
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

    model.eval()

    # Tokenizer
    try:
        tokenizer = Tokenizer(checkpoint_dir, force_backend="huggingface")
    except:
        tokenizer = Tokenizer(checkpoint_dir)

    if args.verb:
        print(f"Using {tokenizer.backend} tokenizer")

    prompt_style = (
        load_prompt_style(checkpoint_dir)
        if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )
    stop_tokens = prompt_style.stop_tokens(tokenizer)

    # ---- GENERATION -------------------------------------------------------------
    # Encode the prompt
    # Run generation
    tok_time_all = []
    with ctx, torch.inference_mode():
        while True:
            prompt = interactive_prompt(prompt_style)
            t_start_msg = time.time()
            start_ids = tokenizer.encode(prompt, device=torch_device)
                # Ensure the desired amount of new tokens is generated
            max_new_tokens = model.max_seq_length
            print(">> Reply: ", end="")

            y = model.generate_chat(
                start_ids,
                max_new_tokens,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                stop_tokens=stop_tokens,
            )
            n_decoded_tok = decode(tokenizer, y, device=torch_device)
            t_msg = time.time() - t_start_msg
            tok_time_all.append((n_decoded_tok, t_msg))
            print("")

            for block in model.transformer.h:
                block.attn.kv_cache.reset_parameters()


if __name__ == "__main__":
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    parser = ArgumentParser(description="""LLM inference""")
    parser.add_argument("-v", "--verb", action="store_true", help="enable verbose mode")
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="if set, compile the model (Torch >= 2.0.0 required)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=script_dir / "checkpoints",
        help=f"folder containing model files (default={script_dir / 'checkpoints'})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help=f"torch device where to load model and tensors (default={default_device}",
    )
    parser.add_argument("--seed", type=int, default=10137, help="set random seed")

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\nGeneration completed!")
