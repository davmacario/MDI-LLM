#!/usr/bin/env python3

import argparse
import os

from .config import (BATCH_SIZE, CKPT_INTERVAL, DEVICE,
                     GRADIENT_ACCUMULATION_STEPS, INIT_FROM, LOG_INTERVAL,
                     MAX_ITERS, PLOTS)

script_dir = os.path.dirname(__file__)


def parse_args(train: bool = True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verb", default=False, action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode (enable profiler)",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        type=str,
        help="Device used to process the model - 'cpu'/'cuda:n'/'mps'",
    )
    if train:
        parser.add_argument("--ckpt", default=None, help="Specify checkpoint file name")
        parser.add_argument(
            "--dataset",
            default=None,
            help="""Name of the data set used; it must be the name of one of the
            subfolders of `data`""",
        )
        parser.add_argument(
            "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
        )
        parser.add_argument(
            "--init",
            type=str,
            default=INIT_FROM,
            help="""Can be: 'scratch', 'resume' or 'gpt2' (and variants: 'gpt2-medium',
            'gpt2-large', 'gpt2-xl') - it decides whether a new model is trained or an
            existing one is used as starting point""",
        )
        parser.add_argument(
            "--max-iters",
            type=int,
            default=MAX_ITERS,
            help="Maximum number of iterations for the training loop (epochs)",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=LOG_INTERVAL,
            help="Number of iterations between each printed log (stdout)",
        )
        parser.add_argument(
            "--ckpt-interval",
            type=int,
            default=CKPT_INTERVAL,
            help="""Number of iterations between each checkpoint (when weights get
            stored)""",
        )
        parser.add_argument(
            "--grad-acc-steps",
            type=int,
            default=GRADIENT_ACCUMULATION_STEPS,
            help="""Gradient accumulation steps - used to simulate larger batch size at
            training""",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=None,
            help="""If specified, set the maximum number of checkpoint intervals with
            non-decreasing loss before interrupting training""",
        )
        parser.add_argument(
            "--always-update",
            default=False,
            action="store_true",
            help="""If this flag is set, the checkpoint will always be updated every
            'ckpt-interval'""",
        )
    else:
        parser.add_argument(
            "-p",
            "--plots",
            default=PLOTS,
            action="store_true",
            help="Produce plots and store points as csv files (./logs/tok-per-time)",
        )
        parser.add_argument(
            "--ckpt",
            default=None,
            help="""Specify checkpoint file name to initialize the model from; if one of
            'gpt2', 'gpt2-medium', 'gpt2-large' or 'gpt2-xl', attempts to load one of
            these pretrained models from Huggingface""",
        )
        # Output file for storing times
        parser.add_argument(
            "--time-run",
            type=str,
            default=None,
            help="""Path of the file where to store the run information and generation
            time""",
        )
        parser.add_argument(
            "--n-tokens",
            type=int,
            default=500,
            help="Maximum number of tokens per sample to be generated",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            default="\n",
            help="""Prompt for generation - can specify a file by name calling
            'FILE:<path>.txt'""",
        )
        parser.add_argument(
            "--n-samples",
            type=int,
            default=3,
            help="Number of samples to be generated",
        )
    return parser.parse_args()
