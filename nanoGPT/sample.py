#!/usr/bin/env python3

"""
Perform inference on a pre-trained model
"""

import cProfile
import os
import pickle
import pstats
import time
from contextlib import nullcontext

import tiktoken
import torch

from sub.char_tokenizer import CharacterTokenizer
from sub.config import (COMPILE, DEVICE, DTYPE, INIT_FROM, TEMPERATURE, TOP_K,
                        VERB)
from sub.model import GPT, GPTConfig
from sub.parser import parse_args

script_dir = os.path.dirname(__file__)

PROFILE = False


def main():
    # --------------------------------------------------------------------------
    dataset = "shakespeare"
    dataset_name = os.path.splitext(dataset)[0]
    data_dir = os.path.join(script_dir, "data", dataset_name)

    args = parse_args(train=False)

    if args.prompt.startswith("FILE:"):
        with open(args.prompt[5:], "r") as f:
            start = f.read()
    else:
        start = args.prompt
    num_samples = args.n_samples  # number of samples to draw
    max_new_tokens = args.n_tokens  # number of tokens generated in each sample
    seed = 1337

    if args.ckpt is not None:
        assert os.path.exists(args.ckpt)
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(data_dir, "out", "ckpt.pt")

    VERB = args.verb
    global PROFILE
    PROFILE = args.debug

    # --------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # for later use in torch.autocast:
    if "cuda" in DEVICE:
        device_type = "cuda"
    elif "mps" in DEVICE:
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

    # model
    # init from a model saved in a specific directory
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."  # NOTE: this shouldn't happen anymore
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    # elif INIT_FROM.startswith("gpt2"):
    #     # init from a given GPT-2 model
    #     model = GPT.from_pretrained(INIT_FROM, dict(dropout=0.0))

    model.to(DEVICE)
    model.eval()
    # if COMPILE:
    #     model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    # Look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    meta_path = None
    if (
        INIT_FROM == "resume"
        and "config" in checkpoint
        and "DATASET" in checkpoint["config"]
    ):  # older checkpoints might not have these...
        dataset_name = os.path.basename(
            os.path.normpath(checkpoint["config"]["DATASET"])
        )
        meta_path = os.path.join(script_dir, "data", dataset_name, "meta.pkl")
        if VERB:
            print("Looking for tokenizer info in: ", meta_path)
        load_meta = os.path.exists(meta_path)

    # Free up memory
    checkpoint = None

    if load_meta and meta_path is not None:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        tok = CharacterTokenizer(meta["stoi"], meta["itos"])
        encode = lambda s: tok.encode(s)
        decode = lambda l: tok.decode(l)
    else:
        # Assume gpt-2 encodings by default FIXME
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # ---- GENERATION -------------------------------------------------------------
    # Encode the beginning of the prompt
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...]

    # Run generation
    with torch.no_grad():
        # with ctx:
        if VERB:
            print("Beginning generation")
        t_start = time.time()
        for k in range(num_samples):
            y = model.generate(
                x, max_new_tokens, temperature=TEMPERATURE, top_k=TOP_K
            )
            print(decode(y[0].tolist()))
            print("---------------")

    if VERB:
        print(f"Total generation time: {time.time() - t_start} s")


if __name__ == "__main__":
    if PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

    main()

    if PROFILE:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats()
        stats.dump_stats(
            os.path.join(script_dir, "logs", "sample_profile.prof")
        )
