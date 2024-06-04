#!/usr/bin/env python3

import argparse
import json
import logging
import os
import warnings

import cherrypy as cp
import torch

from sub.model_dist import GPTServer

# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
settings_path = os.path.join(script_dir, "settings_distr")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk",
    type=str,
    default=None,
    help="""Optional path of the model chunk on disk - if not provided, need to ensure
    the starter node will transmit the model chunk.""",
)
parser.add_argument("-v", "--verb", action="store_true", help="Enable verbose mode")
parser.add_argument(
    "-d",
    "--debug",
    default=False,
    action="store_true",
    help="Enable debug mode (enable profiler)",
)

# The following achieves: (--nodes-config & IND) | --secondary-config
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--nodes-config",
    type=str,
    metavar=("CONFIG-PATH", "SECONDARY-INDEX"),
    nargs=2,  # 2 args
    default=[os.path.join(script_dir, "settings_distr", "configuration.json"), 0],
    help="""Path to the JSON configuration file for the nodes followed by the positional
    index of the intermediate node""",
)
group.add_argument(
    "--secondary-config",
    type=str,
    default=None,
    help="""Path of the configuration for the secondary node, alternative to
    '--nodes-config'""",
)

if __name__ == "__main__":
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # The only available command line option is '--debug' to launch logger
    args = parser.parse_args()

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

    if args.secondary_config is None:
        with open(args.nodes_config[0]) as f:
            full_config = json.load(f)
            n_secondary = len(full_config["nodes"]["secondary"])
            node_ind = int(args.nodes_config[1])
            if node_ind >= n_secondary:
                raise ValueError(
                    f"""Invalid index for the current node: {node_ind} - valid indices
                    are in the range 0 - {n_secondary} for this config file"""
                )
            node_config = full_config["nodes"]["secondary"][node_ind]
    else:
        with open(args.secondary_config) as f:
            node_config = json.load(f)

    if args.chunk is not None and not ("secondary" in args.chunk):
        warnings.warn("Possibly wrong chunk file detected")

    try:
        setup = {"verb": args.verb}  # Override globals in other file
        gpt_webserv = GPTServer(
            node_config=node_config,
            chunk_path=args.chunk,
            **setup,
        )
    except KeyboardInterrupt:
        print("Node stopped!")
        cp.engine.stop()
