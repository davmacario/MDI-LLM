#!/usr/bin/env python3

import argparse
import os

import transformers

from sub.config import (BIAS, BLOCK_SIZE, DROPOUT, N_EMBD, N_HEADS,  # TODO
                        N_LAYER)
from sub.model import GPT, GPTConfig

script_docstring = """
This script is used to split a GPT model into multiple chunks, as specified by the
command line args.

It is possible to split either checkpoints of models that have been trained "locally"
(.pt files stored on disk), or pre-trained models from huggingface.

The produced file chunks will be stored as .pt files on disk, and will include, besides
the piece of the model, all the metadata required to load the chunk and initialize a 
GPTDistributed object.
"""

script_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description=script_docstring)
parser.add_argument(
    "model",
    type=str,
    help="Model to be split. It can either be the path of a model stored locally (.pt) or a gpt2 flavor",
)
parser.add_argument(
    "n_chunks",
    type=int,
    help="Number of chunks to be produced. How the model is split depends on the configuration (config.py)",
)
# Optional:
parser.add_argument(
    "-o",
    "--out-dir",
    type=str,
    help="Directory where to place the '.pt' files containing the chunks obtained by splitting the model",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # TODO:
    # - Open correct model
    # - If .pt: can use the usual function to split model
    # - Else: go through code, my guess is that we should first load the GPT model
    #   'from_pretrained', then extract the state_dict(), on which we can call the usual
    #   function to split the parameters
    # NOTE: need to be careful about the memory consumption - would it be possible to
    # load the model without initializing the class???
