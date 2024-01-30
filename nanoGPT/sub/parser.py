#!/usr/bin/env python3

import argparse

from .config import BATCH_SIZE, INIT_FROM, LOG_INTERVAL, MAX_ITERS


def parse_args():
    parser = argparse.ArgumentParser()
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
        "--dataset",
        type=str,
        default="shakespeare",
        help="Name of the data set used for training",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=MAX_ITERS,
        help="Maximum number of iterations for the training loop",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=LOG_INTERVAL,
        help="Number of iterations between each printed log",
    )
    parser.add_argument(
        "--verb", default=False, action="store_true", help="Enable verbose mode"
    )
    return parser.parse_args()
