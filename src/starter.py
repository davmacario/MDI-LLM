#!/usr/bin/env python3

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import cherrypy as cp
import torch

from sub.model_dist import GPTDistributed

# -----------------------------------------------------------------------------
script_dir = Path(os.path.dirname(__file__))
data_dir = os.path.join(script_dir, "data", "shakespeare_gpt2")

csv_header_stats = ",".join(
    ["timestamp", "n_samples", "n_layers", "context_size", "gen_time"]
)


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    print("+------------------------+")
    print("| Launching starter node |")
    print("+------------------------+")

    tok_per_sample = args.n_tokens
    if args.debug:
        # TODO: review
        log_file = os.path.join(script_dir, "logs", "logs_starter.log")
        log_wp = logging.getLogger("model_dist")
        formatter = logging.Formatter("[%(asctime)s] → %(levelname)s: %(message)s")
        if not os.path.exists(os.path.dirname(log_file)):
            os.mkdir(os.path.dirname(log_file))
        fhdlr = logging.FileHandler(log_file, mode="w")
        fhdlr.setFormatter(formatter)
        log_wp.setLevel(logging.DEBUG)
        log_wp.addHandler(fhdlr)
    out_stats_file = args.time_run
    if out_stats_file is not None:
        assert out_stats_file.parent.is_dir()

    # Init. distributed model, config file from parser
    gpt_distr = GPTDistributed(
        node_type="starter",
        config_file=args.nodes_config,
        ckpt_dir=args.ckpt,
        chunk_path=args.chunk,
        device=args.device,
        model_seq_length=args.sequence_length,
        verb=args.verb,
        plots=args.plots,
    )

    # Operation (start now includes loop)
    gpt_distr.start(
        n_samples=args.n_samples,
        tokens_per_sample=tok_per_sample,
        prompt=args.prompt,
    )
    # # Print the stats to file (we are sure directory exists)
    # if out_stats_file is not None:
    #     # Output csv
    #     existed = True
    #     if not os.path.exists(out_stats_file):
    #         existed = False
    #     with open(out_stats_file, "a") as f:
    #         # Format: datetime - number of samples - model info - total time
    #         curr_ts = datetime.now()
    #         if not existed:
    #             # header
    #             f.write(csv_header_stats + "\n")
    #         f.write(
    #             f"{curr_ts.strftime('%Y-%m-%d %H:%M:%S')},{len(gen_samples)},{gpt_distr.n_layers_tot},{gpt_distr.model_config.block_size},{gen_time}\n"
    #         )
    #         f.close()
    #         print("Stats written to ", out_stats_file)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Starter node - MDI")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable debug mode (profiler)"
    )
    parser.add_argument("-v", "--verb", action="store_true", help="enable verbose mode")
    parser.add_argument("-p", "--plots", action="store_true", help="enable plots")
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="compile Torch module (only for Torch>=2.0.0)",
    )

    parser.add_argument(
        "--ckpt",
        type=Path,
        default=script_dir / "checkpoint",
        help="folder containing the model files",
    )
    parser.add_argument(
        "--chunk", type=Path, default=None, help="path of the model chunk"
    )
    parser.add_argument(
        "--nodes-config",
        type=Path,
        default=Path(os.path.join(script_dir, "settings_distr", "configuration.json")),
        help="""path of the JSON configuration file for the nodes; if not specified, the
        default 'settings_distr/configuration.json' will be used""",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device where to load model and tensors",
    )
    # Run
    parser.add_argument(
        "--prompt",
        type=str,
        default="Who are you?",
        help="""specify a prompt for the language model;
        if starting with 'FILE:', the prompt will be extracted for a file.
        If the string is the prompt itself, it will be used for all generated samples,
        while if a file is specified, each paragraph (separated by blank line) will be 
        used for a different sample, with extra prompts being discarded if --n-samples
        is lower than the number of paragraphs.
        """,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="number of samples (independent pieces of text) to be generated",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=300,
        help="number of tokens to be generated, default: 300",
    )
    parser.add_argument(
        "--sequence-length",
        "--context-length",
        "--block-size",
        type=int,
        default=None,
        help="""
        sequence length of the model, i.e., maximum span of the attention window;
        if not specified, it will use the default model sequence length;
        allows to reduce RAM usage, as with a shorter context less cache is created.
        """
    )
    parser.add_argument(
        "--time-run",
        default=None,
        type=Path,
        help="""optional path of the file where to store the run information and generation
        time""",
    )
    parser.add_argument("--seed", type=int, default=10137, help="set random seed")
    args = parser.parse_args()

    main(args)
