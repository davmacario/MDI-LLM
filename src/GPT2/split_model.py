#!/usr/bin/env python3

import argparse
import gc
import os
import warnings

from sub.utils import (load_from_hf, load_from_pt, remove_prefix,
                       split_parameters)

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
    help="""Model to be split. It can either be the path of a model stored locally
    (.pt) or a gpt2 flavor""",
)
parser.add_argument(
    "n_chunks",
    type=int,
    help="""Number of chunks to be produced. How the model is split depends on the
    configuration (config.py)""",
)
# Optional:
parser.add_argument(
    "-o",
    "--out-dir",
    type=str,
    default=os.path.join(script_dir, "out", "split_models"),
    help="""Directory where to place the '.pt' files containing the chunks obtained by 
    splitting the model; note that any older model present in the same folder will be
    overwritten""",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # NOTE: need to be careful about the memory consumption - would it be possible to
    # load the model without initializing the class???
    valid_gpt2_flavors = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

    if args.n_chunks < 2:
        raise ValueError("The number of nodes must be at least 2")
    n_nodes = args.n_chunks

    input_model = args.model
    if os.path.exists(input_model):
        print(f"Loading model from {input_model}")
        model_type = remove_prefix(os.path.splitext(os.path.basename(input_model))[0][:4], "_")
        # Assuming input model in dataset/out/
        up_1_level = os.path.dirname(input_model)
        dataset_path = os.path.dirname(up_1_level) if os.path.basename(up_1_level) == "out" else up_1_level
        config = {
            "DATASET": os.path.basename(dataset_path),
            "DATASET_PATH": dataset_path,
        }
        state_dict, model_args = load_from_pt(input_model)
    elif input_model in valid_gpt2_flavors:
        print(f"Using GPT2 flavor: {input_model}")
        model_type = input_model
        state_dict, model_args = load_from_hf(input_model)
        config = {}
    else:
        raise ValueError(f"Unable to find model: {input_model}")

    # Divide loaded state dictionary in chunks
    chunks, layer_info = split_parameters(state_dict, n_nodes)
    if len(state_dict) > 0:
        warnings.warn(f"{len(state_dict)} elements have not been used")
    del state_dict
    gc.collect()

    # Print result of chunk partition
    n_interm = n_nodes - 2
    print("Using the following split:")
    print(f"Starter: {layer_info["N_LAYERS_START"]} layers")
    print(f"{n_interm} Intermediate(s): {layer_info["N_LAYERS_INTERM"]} layers")
    print(f"Finisher: {layer_info["N_LAYERS_FINISH"]} layers")

    # Store each chunk as dict:
    # - model: actual model chunk (params)
    # - model_args: model parameters (to be passed to GPTConfig after)
    # - config: globals of training - maybe not needed...
    # - dist_config: layer_info

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Starter
    starter_fname = os.path.join(args.out_dir, f"ckpt_{model_type}_starter.pt")
    ckpt = {
        "model": chunks["starter"],
        "model_args": model_args,
        "dist_config": layer_info,
        "config": config,
    }

    # Intermediate nodes
    for i in range(n_interm):
        curr_fname = os.path.join(args.out_dir, f"ckpt_{model_type}_intermediate{i}.pt")

    # Finisher
    finisher_name = os.path.join(args.out_dir, f"ckpt_{model_type}_finisher.pt")

    print("Files have been written to disk!")
