#!/usr/bin/env python3

import gc
import math
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from numpy.typing import NDArray
from torch import nn
from transformers import GPT2LMHeadModel

from sub.config import (EVAL_ITERS, LEARNING_RATE, LR_DECAY_ITERS, MIN_LR,
                     N_LAYERS_NODES, WARMUP_ITERS)
from sub.model import Config
from sub.utils.data_loader import get_batch

VERB = False


def get_obj_size(obj):
    """
    Get actual size of python object in memory (in bytes)
    """
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {
            o_id: o
            for o_id, o in all_refr
            if o_id not in marked and not isinstance(o, type)
        }

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


@torch.no_grad()  # Tell the program not to evaluate the gradients (no BP)
def estimate_loss(
    model: nn.Module,
    train: Union[torch.Tensor, NDArray],
    val: Union[torch.Tensor, NDArray],
    batch_size: int,
    device: str,
    *args,
    **kwargs,
) -> Dict[str, float]:
    """
    Evaluate the mean loss over a fixed number of iterations during training.
    This allows to remove possible noise and provide more meaningful
    results.

    Args:
        model: the model on which to measure the loss
        train: training data set (tensor)
        val: validation data set (tensor)

    Returns:
        Dict containing the keys:
            "train": mean loss over EVAL_ITERS iterations for training set
            "val": mean loss over EVAL_ITERS iterations for validation set
    """
    ctx = kwargs.get("ctx", nullcontext())

    out = {}
    dss = {
        "train": train,
        "val": val,
    }
    # Set model to evaluation mode
    model.eval()
    for split in dss.keys():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(dss[split], batch_size, device, model.config)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Re-set the model to training mode
    model.train()
    return out


def get_lr(
    it,
    lr: float = LEARNING_RATE,
    min_lr: float = MIN_LR,
    warmup_it: int = WARMUP_ITERS,
    lr_decay_it: int = LR_DECAY_ITERS,
):
    """
    Evaluate learning rate for decayed LR.
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_it:
        return lr * it / warmup_it
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_it:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_it) / (lr_decay_it - warmup_it)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)


def loading_bar(
    current_iter: int,
    tot_iter: int,
    n_chars: int = 10,
    ch: str = "=",
    n_ch: str = " ",
) -> str:
    """
    loading_bar
    ---
    Produce a loading bar string to be printed.

    Args:
        current_iter: current iteration, will determine the position
            of the current bar
        tot_iter: total number of iterations to be performed
        n_chars: total length of the loading bar in characters
        ch: character that makes up the loading bar (default: =)
        n_ch: character that makes up the remaining part of the bar
            (default: blankspace)

    Returns:
        string containing the loading bar for the current iteration
    """
    n_elem = int(current_iter * n_chars / tot_iter)
    prog = str("".join([ch] * n_elem))
    n_prog = str("".join([n_ch] * (n_chars - n_elem - 1)))
    return "[" + prog + n_prog + "]"


def remove_prefix(text: str, prefix: str) -> str:
    """
    Remove the specified prefix from the given string.
    NOTE: starting Python 3.9, use text.removeprefix(prefix);
    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


# FIXME
def find_eot(text: str, stop_tokens: Optional[List[int]] = None) -> int:
    """
    Return the index of the first character of '<|endoftext|>', if found in text.
    Else, return len(text)
    """
    # TODO: add support for tokenizer eos and bos
    tbf = "<|endoftext|>"

    for i in range(0, len(text) - len(tbf)):
        if text[i:].startswith(tbf):
            return i

    return len(text)


def split_parameters(
    model_params: Dict[str, Any], n_nodes: int
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Split the model parameters (contained in a state dict) among the different
    available nodes.

    The number of nodes should be at least 2 (starter and finisher).

    The parameters are divided as such:
        - Starter: token embedding, positional embedding,
            N_LAYERS_STARTxTransformer Layers + final layer norm and linear layer
        - Secondary: N_LAYERS_INTERMxTransformer Layer

    Args:
        model_params: complete model parameters (state dict)
        n_nodes: number of nodes among which to divide the parameters; must be
        greater or equal to 2 (at least starter and finisher)

    Returns:
        dict containing the following k-v pairs:
            "starter": dict with the starter state dict
            "secondary": list containing the intermediate state dicts
        Layers information (n. layers per node)
    """
    assert n_nodes >= 2, "There must be at least 2 nodes in the network"

    # Set up some parameters - they are used to gather the relevant keys
    base_name_transformer = "transformer"  # Name of the ModuleDict in GPT
    tok_emb = "wte"
    pos_emb = "wpe"
    layer_name = "h"
    transformer_last = f"{base_name_transformer}.ln_f"
    output_layer = "lm_head"
    n_secondary_nodes = n_nodes - 1

    len_before = len(model_params)

    # Count the number of detected transformer layers and check consistency
    layer_keys = [
        k
        for k in model_params.keys()
        if k.startswith(f"{base_name_transformer}.{layer_name}")
    ]
    layers_unique = list(set([".".join(k.split(".")[:3]) for k in layer_keys]))
    n_layers_model = len(layers_unique)
    if VERB:
        print(f"Number of transformer layers found in the model: {n_layers_model}")

    layers_info = {}
    n_layers_start = N_LAYERS_NODES[n_nodes][n_layers_model]["N_LAYERS_START"]
    layers_info["N_LAYERS_START"] = n_layers_start
    n_layers_secondary = N_LAYERS_NODES[n_nodes][n_layers_model]["N_LAYERS_SECONDARY"]
    layers_info["N_LAYERS_SECONDARY"] = n_layers_secondary

    if VERB:
        print(f"Number of layers - starter node: {n_layers_start}")
        print(
            f"Number of layers - secondary node{'s' if n_layers_secondary > 1 else ''}: {n_layers_secondary}"
        )

    out_chunks = {}

    # 1. Select params for Starter
    out_chunks["starter"] = {}
    out_chunks["starter"][f"starter_model.{tok_emb}.weight"] = model_params.pop(
        f"{base_name_transformer}.{tok_emb}.weight"
    )
    out_chunks["starter"][f"starter_model.{pos_emb}.weight"] = model_params.pop(
        f"{base_name_transformer}.{pos_emb}.weight"
    )
    if f"{base_name_transformer}.{tok_emb}.bias" in model_params.keys():
        out_chunks["starter"][f"starter_model.{tok_emb}.bias"] = model_params.pop(
            f"{base_name_transformer}.{tok_emb}.bias"
        )

    if f"{base_name_transformer}.{pos_emb}.bias" in model_params.keys():
        out_chunks["starter"][f"starter_model.{pos_emb}.bias"] = model_params.pop(
            f"{base_name_transformer}.{pos_emb}.bias"
        )

    # Starter transformer layers
    # Complicated pythonic list call to select the correct keys to be transferred to the
    # starter node
    # As reference, the keys for the layers all start with:
    #               transformer.h.<layer_ind>.[...]
    # so we need to select the correct layer indices
    valid_layer_ind = list(range(0, n_layers_start))
    relevant_keys = [  # Keys of the original model that will be copied
        k
        for k in list(model_params.keys())
        if (
            k.startswith(f"{base_name_transformer}.{layer_name}")
            and int(k.split(".")[2]) in valid_layer_ind
        )
    ]

    for k_orig in relevant_keys:
        ind_layer = int(k_orig.split(".")[2])
        ind_layer_chunk = ind_layer  # Starter layers will have the same index

        prefix = f"{base_name_transformer}.{layer_name}.{ind_layer}."
        end = remove_prefix(k_orig, prefix)
        new_k = f"starter_model.{layer_name}.{ind_layer_chunk}.{end}"
        out_chunks["starter"][new_k] = model_params.pop(k_orig)

    # ln_f - last layernorm
    out_chunks["starter"][f"starter_model.ln_f.weight"] = model_params.pop(
        f"{transformer_last}.weight"
    )
    if f"{transformer_last}.bias" in model_params.keys():
        out_chunks["starter"][f"starter_model.ln_f.bias"] = model_params.pop(
            f"{transformer_last}.bias"
        )

    # lm_head - final linear layers (producing PMF over tokenizer vocabulary)
    out_chunks["starter"][f"starter_model.lm_head.weight"] = model_params.pop(
        f"{output_layer}.weight"
    )
    if f"{output_layer}.bias" in model_params.keys():
        out_chunks["starter"][f"starter_model.lm_head.bias"] = model_params.pop(
            f"{output_layer}.bias"
        )

    # 2. Select params for every Secondary
    out_chunks["secondary"] = []
    for i in range(1, n_nodes):
        curr_params = {}

        # Calculate valid layers indices in the original model
        start_layer_ind = n_layers_start + (i - 1) * n_layers_secondary
        finish_layer_ind = n_layers_start + i * n_layers_secondary
        valid_layer_ind = list(range(start_layer_ind, finish_layer_ind))
        relevant_keys = [
            k
            for k in list(model_params.keys())
            if (
                k.startswith(f"{base_name_transformer}.{layer_name}")
                and int(k.split(".")[2]) in valid_layer_ind
            )
        ]

        for k_orig in relevant_keys:
            ind_layer = int(k_orig.split(".")[2])
            ind_layer_chunk = ind_layer - start_layer_ind

            prefix = f"{base_name_transformer}.{layer_name}.{ind_layer}."
            end = remove_prefix(k_orig, prefix)
            new_k = f"secondary_model.{layer_name}.{ind_layer_chunk}.{end}"
            curr_params[new_k] = model_params.pop(k_orig)

        out_chunks["secondary"].append(curr_params)

    return out_chunks, layers_info


def serialize_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Serialize a mapping, specifically a state dict, to allow it to be read as a
    JSON/dict.
    """
    json_serializable_params = {}
    for key, value in params.items():
        json_serializable_params[key] = (
            value.tolist() if isinstance(value, torch.Tensor) else value
        )

    return json_serializable_params


def deserialize_params(params: Dict) -> Mapping[str, Any]:
    """
    De-serialize a dictionary and return a state dict containing torch model parameters.
    """
    deserialized_params = {}
    for key, value in params.items():
        if isinstance(value, list):
            # Convert lists back to PyTorch tensors
            deserialized_params[key] = torch.tensor(value)
        else:
            deserialized_params[key] = value

    return deserialized_params


def count_transformer_blocks(state_dict: Dict[str, Any]) -> int:
    """
    Given a state dict, return the number of detected transformer blocks
    """
    base_name_transformer = "transformer"
    layer_name = "h"

    # Count the number of detected transformer layers
    layer_keys = [
        k
        for k in state_dict.keys()
        if k.startswith(f"{base_name_transformer}.{layer_name}")
    ]
    layers_unique = list(set([".".join(k.split(".")[:3]) for k in layer_keys]))
    return len(layers_unique)


def load_from_pt(
    model_path: Union[Path, str], device: Optional[Union[torch.device, str]] = "cpu"
) -> Tuple[Config, Dict[str, Any]]:
    """
    Load model weights from disk.

    Args:
        model_path: path to the checkpoint
        device (optional): device where to load state dict; default: "cpu"

    Returns:
        tuple (model config, model state dict)
    """
    if isinstance(model_path, str):
        model_dir = Path(model_path)
    elif isinstance(model_path, Path):
        model_dir = model_path
    else:
        raise TypeError

    if not model_dir.is_dir():
        raise NotADirectoryError(f"Unable to find model checkpoint at {model_dir}")

    config = Config.from_file(model_dir / "model_config.yaml")

    pth_file = model_dir / "lit_model.pth"
    try:
        sd = torch.load(pth_file, map_location=device)
    except:
        if device != "cpu":
            warnings.warn(
                f"Unable to fit model ckpt in {device} memory! Retrying with cpu"
            )
            sd = torch.load(pth_file, map_location="cpu")
        else:
            raise MemoryError("Not enough system memory to load ckpt!")

    return config, sd


def load_from_hf(model_type: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load model weights from Huggingface.

    Args:
        model_type: one of ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

    Returns:
        model state dictionary, imported from gpt2
        model config (arguments of GPTConfig) as dictionary
    """
    pass