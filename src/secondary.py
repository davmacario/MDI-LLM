#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path

import torch

from sub import App

# -----------------------------------------------------------------------------
script_dir = Path(os.path.dirname(__file__))
settings_path = Path(os.path.join(script_dir, "settings_distr"))


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    print("+---------------------------------+")
    print("| Launching secondary worker node |")
    print("+---------------------------------+")

    if args.debug:
        log_file = os.path.join(script_dir, "logs", "logs_finisher.log")
        log_wp = logging.getLogger("model_dist")
        formatter = logging.Formatter("[%(asctime)s] → %(levelname)s: %(message)s")
        if not os.path.exists(os.path.dirname(log_file)):
            os.mkdir(os.path.dirname(log_file))
        fhdlr = logging.FileHandler(log_file, mode="w")
        log_wp.setLevel(logging.DEBUG)
        fhdlr.setFormatter(formatter)
        log_wp.addHandler(fhdlr)

    role = "secondary"

    app = App(
        node_type=role,
        node_config=args.node_config,
        device=args.device,
        dtype=args.dtype,
        verb=args.verb,
    )

    app.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable debug mode (profiler)"
    )
    parser.add_argument("-v", "--verb", action="store_true", help="enable verbose mode")
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="compile Torch module (only for Torch>=2.0.0)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=script_dir / "checkpoints",
        help="""path of the checkpoints directory (where all models are placed).""",
    )
    parser.add_argument(
        "--node-config",
        type=Path,
        default=script_dir / "settings_distr" / "secondary" / "node0_local.json",
        help="""path to the JSON configuration file for the secondary node.""",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device where to load model and tensors",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="""the model dtype (among float32, float16 and bfloat16 - if supported)""",
    )
    parser.add_argument("--seed", type=int, default=10137, help="set random seed")
    args = parser.parse_args()

    main(args)
