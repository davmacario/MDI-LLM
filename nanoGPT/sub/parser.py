#!/usr/bin/env python3

import argparse
import os

from .config import (BATCH_SIZE, CKPT_INTERVAL, INIT_FROM, LOG_INTERVAL,
                     MAX_ITERS)

script_dir = os.path.dirname(__file__)


def parse_args(train: bool = True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(script_dir, "..", "data", "shakespeare"),
        help="Path of the data set folder used for training; it must be the name of one of the subfolders of `data`",
    )
    parser.add_argument(
        "--verb", default=False, action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--ckpt", default=None, help="Specify checkpoint file name"
    )
    # TODO: add args spec for non-training
    if train:
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
    return parser.parse_args()
