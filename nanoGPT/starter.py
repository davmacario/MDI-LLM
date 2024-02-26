#!/usr/bin/env python3

import logging
import os
from datetime import datetime

import cherrypy as cp
import torch

from sub.config import PLOTS, VERB
from sub.model_dist import GPTDistributed
from sub.parser import parse_args

# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

if __name__ == "__main__":
    # Parse command line arguments
    # Example usage:
    #   python3 nanoGPT/starter.py --dataset=./nanoGPT/data/shakespeare --ckpt=./nanoGPT/data/shakespeare/out/ckpt_5layers.py --debug
    args = parse_args(train=False)

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
        log_file = os.path.join(script_dir, "logs", "logs_starter.log")
        log_wp = logging.getLogger("model_dist")
        formatter = logging.Formatter(
            "[%(asctime)s] → %(levelname)s: %(message)s"
        )
        if not os.path.exists(os.path.dirname(log_file)):
            os.mkdir(os.path.dirname(log_file))
        fhdlr = logging.FileHandler(log_file, mode="w")
        fhdlr.setFormatter(formatter)
        log_wp.setLevel(logging.DEBUG)
        log_wp.addHandler(fhdlr)

    VERB = args.verb
    PLOTS = args.plots

    setup = {"verb": VERB, "plots": PLOTS}

    out_stats_file = args.time_run
    if out_stats_file is not None:
        assert os.path.exists(os.path.dirname(out_stats_file))

    # Init. distributed model, config file from parser
    gpt_distr = GPTDistributed(
        ckpt_path, nodes_info_path=args.nodes_config, **setup
    )

    # Operation
    try:
        gen_samples, gen_time = gpt_distr.start(tokens_per_sample=1000)
    except KeyboardInterrupt:
        cp.engine.stop()
        print("Starter node was stopped successfully!")
    else:
        # Print the stats to file (we are sure directory exists)
        if out_stats_file is not None:
            # Output csv
            existed = True
            if not os.path.exists(out_stats_file):
                existed = False
            with open(out_stats_file, "a") as f:
                # Format: datetime - number of samples - model info - total time
                curr_ts = datetime.now()
                if not existed:
                    # header
                    f.write(
                        ",".join(
                            [
                                "timestamp",
                                "n_samples",
                                "n_layers",
                                "context_size",
                                "gen_time",
                            ]
                        )
                        + "\n"
                    )
                f.write(
                    f"{curr_ts.strftime('%Y-%m-%d %H:%M:%S')},{len(gen_samples)},{gpt_distr.n_layers_tot},{gpt_distr.model_config.block_size},{gen_time}\n"
                )
                f.close()
                print("Stats written to ", out_stats_file)
