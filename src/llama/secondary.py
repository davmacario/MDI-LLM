#!/usr/bin/env python3

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import cherrypy as cp
import torch
from sub import GPTDistributed

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

    role = f"secondary:{args.nodes_config[1]}"
    config_path = Path(args.nodes_config[0])

    gpt_distr = GPTDistributed(
        node_type=role,
        config_file=config_path,
        chunk_path=args.chunk,
        device=args.device,
        verb=args.verb,
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable debug mode (profiler)"
    )
    parser.add_argument("-v", "--verb", action="store_true", help="enable verbose mode")

    parser.add_argument(
        "--chunk",
        type=Path,
        default=None,
        help="""optional path of the model chunk on disk - if not provided, need to ensure
        the starter node will transmit the model chunk.""",
    )
    # The following achieves: (--nodes-config & IND) | --secondary-config
    parser.add_argument(
        "--nodes-config",
        type=str,
        metavar=("CONFIG-PATH", "SECONDARY-INDEX"),
        nargs=2,  # 2 args total
        default=[os.path.join(script_dir, "settings_distr", "configuration.json"), 0],
        help="""path to the JSON configuration file for the nodes followed by the
        positional index of the intermediate node""",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="torch device where to load model and tensors",
    )
    parser.add_argument("--seed", type=int, default=10137, help="set random seed")
    args = parser.parse_args()

    main(args)
