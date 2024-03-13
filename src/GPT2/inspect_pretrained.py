#!/usr/bin/env python3

import argparse
import os

import tiktoken
from transformers import GPT2LMHeadModel

from sub.model import GPT, GPTConfig

script_dir = os.path.dirname(__file__)
my_keys_file = os.path.join(script_dir, "tmp", "my_gpt_keys.txt")
hf_keys_file = os.path.join(script_dir, "tmp", "hf_gpt_keys.txt")

parser = argparse.ArgumentParser()
parser.add_argument(
    "model",
    type=str,
    help="Model to be inspected. It can either be the path of a model stored locally (.pt) or a gpt2 flavor",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if os.path.exists(args.model):
        print("Not implemented")
    elif args.model in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}:
        print(f"Loading pretrained model: {args.model}")
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[args.model]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        with open(my_keys_file, "w") as f:
            print(f"Writing keys of GPT model to {my_keys_file}")
            for k in sd:
                f.write(f"{k}\n")
        # Not excluding any keys (no matches with ".attn.bias")
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # Init. HF model
        model_hf = GPT2LMHeadModel.from_pretrained(args.model)
        sd_hf = model_hf.state_dict()
        with open(hf_keys_file, "w") as f:
            print(f"Writing keys of Huggingface GPT model to {hf_keys_file}")
            for k in sd_hf:
                f.write(f"{k}\n")
        # No matches for "masked_bias", but each layer contains ".attn.bias"
        sd_keys_hf = [
            k
            for k in sd_hf.keys()
            if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))
        ]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys) == len(
            sd_keys_hf
        ), f"HF: {len(sd_keys_hf)} keys, Mine: {len(sd_keys)} keys"

        # TODO: import parameters

    else:
        raise argparse.ArgumentError(args.model, f"Invalid model: {args.model}")
