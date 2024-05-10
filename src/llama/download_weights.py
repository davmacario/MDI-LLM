#!/usr/bin/env python3

import os
from argparse import ArgumentParser
from pathlib import Path

from sub.utils import download_from_hub


def main(args):
    """
    Steps:
    - Download from hub (may need to provide API key for Meta models)
    """

    download_from_hub(
        repo_id=args.model_name,
        access_token=args.hf_token if args.hf_token is not None else os.getenv("HF_TOKEN"),
        dtype=args.dtype,
        checkpoint_dir=args.ckpt_dir,
        model_name=args.saved_name,
        convert_checkpoint=args.no_convert,
    )
    print("Weights saved!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Download model weights from Huggingface Hub")
    parser.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Name of the model on Huggingface Hub",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        # default=(
        #     "bfloat16"
        #     if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        #     else "float16"
        # ),
        default=None,
        help="Data type of downloaded weights - they will be quantized if necessary",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Huggingface token to access restricted/private workspaces",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("checkpoints"),
        help="""Directory where the model will be downloaded; it will be created if not
        present"""
    )
    parser.add_argument(
        "--saved-name",
        type=str,
        default=None,
        help="""If specified, the name of the subfolder of '--ckpt-dir' where the model
        will be stored"""
    )
    parser.add_argument(
        "--no-convert",
        default=True,
        action="store_false",
        help="If set, prevent converting the weights to LitGPT format"
    )
    args = parser.parse_args()
    main(args)
