#!/usr/bin/env python3

import argparse
import dataclasses
import io
import os
from pathlib import Path

import torch
from sub import Config
from sub.utils import count_transformer_blocks
from transformers import AutoConfig, AutoModel

script_dir = Path(os.path.dirname(__file__))
pt_local_file = script_dir / "tmp" / "local_pt_keys.txt"
my_keys_file = script_dir / "tmp" / "my_llama_keys.txt"
hf_keys_file = script_dir / "tmp" / "hf_llama_keys.txt"

if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_methods(object, spacing=20):
    methodList = []
    for method_name in dir(object):
        try:
            if callable(getattr(object, method_name)):
                methodList.append(str(method_name))
        except Exception:
            methodList.append(str(method_name))
    processFunc = (lambda s: " ".join(s.split())) or (lambda s: s)
    for method in methodList:
        try:
            print(
                str(method.ljust(spacing))
                + " "
                + processFunc(str(getattr(object, method).__doc__)[0:90])
            )
        except Exception:
            print(method.ljust(spacing) + " " + " getattr() failed")


def main(args):
    checkpoint_dir = Path(args.model)
    if checkpoint_dir.is_dir():
        conf_file = checkpoint_dir / "model_config.yaml"
        config = Config.from_file(conf_file)
        config_dict = config.asdict()
        for k, v in config_dict.items():
            print(f"{k}: {v}")
        print("")

        pth_file = checkpoint_dir / "lit_model.pth"
        sd = torch.load(pth_file)

        n_blocks_detect = count_transformer_blocks(sd)
        print(f"{n_blocks_detect} transformer blocks detected", end=" ")
        if n_blocks_detect == config_dict["n_layer"]:
            print("-> Config checks out")
        else:
            print("")
            raise ValueError(
                f"{n_blocks_detect} layers have been detected, "
                f"but the configuration says {config_dict['n_layer']}"
            )
    else:
        # TODO: remake
        # Attempt to load from HF
        print(f"Loading pretrained model: {args.model}")
        # model_hf = LlamaPreTrainedModel.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)  # LlamaConfig
        print(config)
        model_hf = AutoModel.from_pretrained(args.model)  # LlamaModel
        get_methods(model_hf)

        sd_hf = model_hf.state_dict()

        buf = io.BytesIO()
        torch.save(sd_hf, buf)
        buf.seek(0)
        print(f"Total HF model size (torch load to buffer): {len(buf.read())} B")

        with open(hf_keys_file, "w") as f:
            print(f"Writing keys of Huggingface model to {hf_keys_file}")
            for k in sd_hf:
                f.write(f"{k}\n")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Inspect pretrained model checkpoints"""
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save model parameter names to default location",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=model,
        help="""The model to be inspected; can be a local folder (containing the
        lit_model.pth and model_config.yaml files) or a Huggingface model""",
    )
    parser.add_argument(
        "--device", type=str, default=device, help="Device where to load models"
    )
    args = parser.parse_args()

    main(args)
