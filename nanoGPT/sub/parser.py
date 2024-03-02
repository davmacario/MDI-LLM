#!/usr/bin/env python3

import argparse
import os

from .config import (BATCH_SIZE, CKPT_INTERVAL, INIT_FROM, LOG_INTERVAL,
                     MAX_ITERS, PLOTS)

script_dir = os.path.dirname(__file__)


def parse_args(train: bool = True, mdi: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path of the data set folder used for training; it must be the name of one of the subfolders of `data`",
    )
    parser.add_argument(
        "--verb", default=False, action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode (enable profiler)",
    )
    parser.add_argument(
        "--ckpt", default=None, help="Specify checkpoint file name"
    )
    if train:
        parser.add_argument(
            "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
        )
        parser.add_argument(
            "--init",
            type=str,
            default=INIT_FROM,
            help="Can be: 'scratch', 'resume' - it decides whether a new model is trained or an existing one is used as starting point",
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
            help="Number of iterations between each checkpoint (when weights get stored)",
        )
    else:
        # Output file for storing times
        parser.add_argument(
            "--time-run",
            type=str,
            default=None,
            help="Path of the file where to store the run information and generation time",
        )
        parser.add_argument(
            "--n-tokens",
            type=int,
            default=1000,
            help="Maximum number of tokens per sample to be generated",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            default="\n",
            help="Prompt for generation - can specify a file by name calling 'FILE:<path>.txt'",
        )
        parser.add_argument(
            "--plots",
            default=PLOTS,
            action="store_true",
            help="Produce plots",
        )
        if not mdi:
            parser.add_argument(
                "--n-samples",
                type=int,
                default=3,
                help="Number of samples to be generated",
            )
        else:
            parser.add_argument(
                "--nodes-config",
                type=str,
                default=os.path.join(
                    script_dir, "..", "settings_distr", "configuration.json"
                ),
                help="Path to the JSON configuration file for the nodes",
            )
    return parser.parse_args()
