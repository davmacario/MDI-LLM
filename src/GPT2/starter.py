#!/usr/bin/env python3

import argparse
import logging
import os
from datetime import datetime

import cherrypy as cp
import torch

from sub.model_dist import GPTDistributed

# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, "data", "shakespeare_gpt2")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

csv_header_stats = ",".join(
    ["timestamp", "n_samples", "n_layers", "context_size", "gen_time"]
)

parser = argparse.ArgumentParser(description="Starter node - MDI")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--model",
    type=str,
    default=None,
    help="""Path/name of the pretrained model. If it is a GPT-2 flavor ('gpt2',
    'gpt2-medium', 'gpt2-large', 'gpt2-xl') it will be downloaded from Huggingface.
    Can be overrode if specifying a chunk (see '-c'/'--chunk')""",
)
group.add_argument(
    "--chunk",
    type=str,
    default=None,
    help="""Path of the chunk of model assigned to the starter node.
    This argument overrides '--model' and will require the other nodes to be provided
    the path of their own chunks as well.""",
)
parser.add_argument(
    "-d",
    "--debug",
    default=False,
    action="store_true",
    help="Enable debug mode (profiler)",
)
parser.add_argument(
    "-v", "--verb", default=False, action="store_true", help="Enable verbose mode"
)
parser.add_argument(
    "-p",
    "--plots",
    default=False,
    action="store_true",
    help="Produce plots and store the points as csv files ('/logs/tok_per_time' folder)",
)
parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    help="""(Optional) prompt string or 'FILE:<path_to_file.txt>' indicating a file where each
    paragraph is a prompt""",
)
parser.add_argument(
    "--n-samples",
    type=int,
    default=3,
    help="Number of samples (independent pieces of text) to be generated",
)
parser.add_argument(
    "--n-tokens",
    type=int,
    default=300,
    help="Number of tokens to be generated, default: 300",
)
parser.add_argument(
    "--time-run",
    default=None,
    type=str,
    help="""Optional path of the file where to store the run information and generation
    time""",
)
parser.add_argument(
    "--nodes-config",
    type=str,
    default=os.path.join(script_dir, "settings_distr", "configuration.json"),
    help="""Path of the JSON configuration file for the nodes; if not specified, the
    default 'settings_distr/configuration.json' will be used""",
)

if __name__ == "__main__":
    # Parse command line arguments
    args = parser.parse_args()

    # NOTE: model and chunk are mutually exclusive (via argparse)!
    if args.model is not None:
        if os.path.exists(args.model):
            ckpt_path = args.model
            out_dir = os.path.dirname(args.model)
            model_is_split = False
        elif args.model in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}:
            ckpt_path = args.model
            out_dir = os.path.dirname(args.model)
            model_is_split = False
        else:
            raise ValueError(f"Unknown pretrained model: {args.model}")
    elif args.chunk is not None:
        if os.path.exists(args.chunk):
            ckpt_path = args.chunk
            out_dir = os.path.dirname(args.chunk)
            model_is_split = True
        else:
            raise FileNotFoundError(f"Unable to find chunk at {args.chunk}")
    else:
        raise ValueError

    settings_path = os.path.join(script_dir, "settings_distr")
    network_conf_path = os.path.join(settings_path, "configuration.json")

    if args.debug:
        log_file = os.path.join(script_dir, "logs", "logs_starter.log")
        log_wp = logging.getLogger("model_dist")
        formatter = logging.Formatter("[%(asctime)s] → %(levelname)s: %(message)s")
        if not os.path.exists(os.path.dirname(log_file)):
            os.mkdir(os.path.dirname(log_file))
        fhdlr = logging.FileHandler(log_file, mode="w")
        fhdlr.setFormatter(formatter)
        log_wp.setLevel(logging.DEBUG)
        log_wp.addHandler(fhdlr)

    tok_per_sample = args.n_tokens

    setup = {"verb": args.verb, "plots": args.plots}

    out_stats_file = args.time_run
    if out_stats_file is not None:
        assert os.path.exists(os.path.dirname(out_stats_file))

    # Init. distributed model, config file from parser
    gpt_distr = GPTDistributed(
        ckpt_path,
        nodes_info_path=args.nodes_config,
        model_was_split=model_is_split,
        **setup,
    )

    # Operation
    try:
        gen_samples, gen_time = gpt_distr.start(
            n_samples=args.n_samples,
            tokens_per_sample=tok_per_sample,
            prompt=args.prompt,
        )
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
                    f.write(csv_header_stats + "\n")
                f.write(
                    f"{curr_ts.strftime('%Y-%m-%d %H:%M:%S')},{len(gen_samples)},{gpt_distr.n_layers_tot},{gpt_distr.model_config.block_size},{gen_time}\n"
                )
                f.close()
                print("Stats written to ", out_stats_file)
