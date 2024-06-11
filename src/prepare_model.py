#!/usr/bin/env python3

import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

from sub.utils import load_from_hf, load_from_pt, split_and_store
from sub.utils.convert_hf_checkpoint import convert_hf_checkpoint

docstring = """
Use this script to:
- Download weights, config and tokenizer info from Huggingface Hub (if not already downloaded)
- Store them in a local folder
- Partition them among a number of nodes, if specified
- Store the partitions at a specific location

Given the model name (required) and the checkpoint folder (optional - default:
`./checkpoints`), the model will be stored at:

    ./<checkpoint folder>/<hf model name>/

and the chunks will be stored in:

    ./<checkpoint folder>/<hf model name>/chunks/<N>nodes/

where `N` is the number of nodes for the partition contained in that subfolder.

---
"""

script_dir = os.path.dirname(__file__)


def main(args):
    os.makedirs(args.ckpt_folder, exist_ok=True)

    if Path(args.MODEL).is_dir():
        if not (args.MODEL / "lit_model.pth").exists() or not (
            args.MODEL / "model_config.yaml"
        ).exists():
            # Need to convert the model to the Lit format
            convert_hf_checkpoint(checkpoint_dir=args.MODEL)

        _, state_dict = load_from_pt(args.MODEL, args.device)
        model_path = Path(args.MODEL)
    else:
        _, state_dict = load_from_hf(
            repo_id=args.MODEL,
            access_token=(
                args.hf_token if args.hf_token is not None else os.getenv("HF_TOKEN")
            ),
            dtype=args.dtype,
            checkpoint_dir=args.ckpt_folder,
            model_name=args.model_name,
            device=args.device,
        )
        model_path = Path(args.ckpt_folder) / args.MODEL

    print("Model was loaded!")

    # Split the model
    if not args.n_nodes:
        return

    assert state_dict is not None
    chunks_subfolder = split_and_store(state_dict, args.n_nodes, model_path, verb=True)

    print(f"Done! The chunks have been written to {chunks_subfolder}")


if __name__ == "__main__":
    parser = ArgumentParser(description=docstring, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "MODEL",
        type=str,
        help="""model to be downloaded - it should correspond to a local folder
        containing a model or to a Huggingface Hub model;""",
    )

    parser.add_argument(
        "--ckpt-folder",
        type=Path,
        default=Path(os.path.join(script_dir, "checkpoints")),
        help="""subfolder where the model directory will be placed; the model files
        will be found at `<ckpt_folder>/<hf_model_name>/`""",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="""allows to specify a different config name to use for this MODEL,
        allowing to download alternative weights for the same architecture""",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        help="""number of nodes among which to partition the model - if not specified,
        the partition will not be performed""",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="""Huggingface Hub token to access restricted/private workspaces;
        not required if the HF_TOKEN env variable is set.""",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="data type of downloaded weights - they will be quantized if necessary",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device where to load model and tensors (default: cpu)",
    )

    args = parser.parse_args()
    main(args)
