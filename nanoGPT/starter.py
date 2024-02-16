#!/usr/bin/env python3

import logging
import os

import cherrypy as cp
import torch

from sub.model_dist import GPTDistributed
from sub.parser import parse_args

# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
log_file = os.path.join(script_dir, "logs", "logs_starter.log")
if not os.path.exists(os.path.dirname(log_file)):
    os.mkdir(os.path.dirname(log_file))
log_wp = logging.getLogger("model_dist")
fhdlr = logging.FileHandler(log_file, mode="w")
formatter = logging.Formatter("[%(asctime)s] → %(levelname)s: %(message)s")
fhdlr.setFormatter(formatter)
log_wp.addHandler(fhdlr)

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

if __name__ == "__main__":
    # Parse command line arguments
    # Example usage:
    #   python3 nanoGPT/starter.py --dataset=./nanoGPT/data/shakespeare --ckpt=./nanoGPT/data/shakespeare/out/ckpt_5layers.py --debug
    args = parse_args()

    if args.dataset is not None:
        assert os.path.isdir(args.dataset)
        data_dir = args.dataset
    else:
        data_dir = os.path.join(script_dir, "data", "shakespeare")

    if args.ckpt is not None:
        assert os.path.exists(args.ckpt)
        ckpt_path = args.ckpt
        out_dir = os.path.dirname(args.ckpt)
    else:
        out_dir = os.path.join(data_dir, "out")
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        # ckpt_path = os.path.join(out_dir, "ckpt_5layers.pt")

    settings_path = os.path.join(script_dir, "settings_distr")
    network_conf_path = os.path.join(settings_path, "configuration.json")

    if args.debug:
        log_wp.setLevel(logging.DEBUG)
    else:
        log_wp.setLevel(logging.INFO)

    gpt_distr = GPTDistributed(ckpt_path)

    # Operation
    try:
        gpt_distr.start(tokens_per_sample=1000)
    except KeyboardInterrupt:
        cp.engine.stop()
        print("Starter node was stopped successfully!")
