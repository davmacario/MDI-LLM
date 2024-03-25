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
    "IND", type=int, help="Index of the secondary node to be launched on this host"
)
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
parser.add_argument(
    "--nodes-config",
    type=str,
    default=os.path.join(script_dir, "settings_distr", "configuration.json"),
    help="Path to the JSON configuration file for the nodes",
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

    network_conf_path = args.nodes_config

    if args.chunk is not None and not ("secondary" in args.chunk):
        warnings.warn("Possibly wrong chunk file detected")

    try:
        with open(network_conf_path, "r") as f:
            full_config = json.load(f)
            n_secondary_conf = len(full_config["nodes"]["secondary"])
            if args.IND >= n_secondary_conf:
                raise ValueError(
                    f"Invalid index for the current node: {args.IND} - valid indices are in the range 0 - {n_secondary_conf} for this config file"
                )
            setup = {"verb": args.verb}
            gpt_webserv = GPTServer(
                node_config=full_config["nodes"]["secondary"][args.IND],
                chunk_path=args.chunk,
                **setup,
            )
    except KeyboardInterrupt:
        print("Node stopped!")
        cp.engine.stop()