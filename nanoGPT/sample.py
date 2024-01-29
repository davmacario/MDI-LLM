#!/usr/bin/env python3

"""
Perform inference on a pre-trained model
"""

import os
import pickle

# import tiktoken
import torch

from sub.config import COMPILE, DEVICE, DTYPE, INIT_FROM, TOP_K, VERB
from sub.model import GPT, GPTConfig

# from contextlib import nullcontext


# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
# out_dir = os.path.join(script_dir, "out")  # Checkpoints

dataset = "shakespeare"
dataset_name = os.path.splitext(dataset)[0]
data_dir = os.path.join(script_dir, "data", dataset_name)
out_dir = os.path.join(data_dir, "out")

# TODO: write configurator - command line arg parser to overwrite configuration
# parameters

start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 1000  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
seed = 1337
# exec(open("configurator.py").read())  # overrides from command line or config file

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# for later use in torch.autocast:
if "cuda" in DEVICE:
    print("Using GPU")
    device_type = "cuda"
elif "mps" in DEVICE:
    print("Using MPS")
    device_type = "mps"
else:
    print("Using CPU")
    device_type = "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[DTYPE]
# ctx = (
#     nullcontext()
#     if device_type in {"cpu", "mps"}
#     else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# )

# model
if VERB:
    print("Initialization")
if INIT_FROM == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
# elif INIT_FROM.startswith("gpt2"):
#     # init from a given GPT-2 model
#     model = GPT.from_pretrained(INIT_FROM, dict(dropout=0.0))
else:
    raise ValueError(f"Unknown initialization: {INIT_FROM}")

model.eval()
model.to(DEVICE)
# if COMPILE:
#     model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# Look for the meta pickle in case it is available in the dataset folder
if VERB:
    print("Looking for tokenizer metadata")
load_meta = False
if (
    INIT_FROM == "resume"
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):  # older checkpoints might not have these...
    meta_path = os.path.join(
        script_dir, "data", checkpoint["config"]["dataset"], "meta.pkl"
    )
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    raise ValueError
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# ---- GENERATION -------------------------------------------------------------
# Encode the beginning of the prompt
if VERB:
    print("Starting generation")
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...]

# Run generation
with torch.no_grad():
    # with ctx:
    for k in range(num_samples):
        y = model.generate(
            x, max_new_tokens, temperature=temperature, top_k=TOP_K
        )
        print(decode(y[0].tolist()))
        print("---------------")
