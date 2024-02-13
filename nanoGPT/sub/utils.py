#!/usr/bin/env python3

import math
from typing import Any, Dict, Mapping, Union

import torch
from numpy.typing import NDArray
from torch import nn

from .config import (EVAL_ITERS, LEARNING_RATE, LR_DECAY_ITERS, MIN_LR,
                     N_LAYERS_FINISH, N_LAYERS_INTERM, N_LAYERS_START,
                     WARMUP_ITERS)
from .data_loader import get_batch
from .model import GPT


@torch.no_grad()  # Tell the program not to evaluate the gradients (no BP)
def estimate_loss(
    model: Union[GPT, nn.Module],
    train: Union[torch.Tensor, NDArray],
    val: Union[torch.Tensor, NDArray],
):
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
    out = {}
    dss = {
        "train": train,
        "val": val,
    }
    # Set model to evaluation mode
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(dss[split], model.config)
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
    n_prog = str("".join([" "] * (n_chars - n_elem - 1)))
    return "[" + prog + n_prog + "]"


def remove_prefix(text: str, prefix: str) -> str:
    """
    Remove the specified prefix from the given string.
    NOTE: starting Python 3.9, use text.removeprefix(prefix)
    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def split_parameters(
    model_params: Mapping[str, Any], n_nodes: int
) -> Dict[str, Any]:
    """
    Split the model parameters (contained in a state dict) among the different
    available nodes.

    The number of nodes should be at least 2 (starter and finisher).

    The parameters are divided as such:
        - Starter: token embedding, positional embedding
        - Intermediate: 2xTransformer Layer
        - Finisher: 2xTransformer Layer, LayerNorm

    Args:
        model_params: complete model parameters (state dict)
        n_nodes: number of nodes among which to divide the parameters; must be
        greater or equal to 2 (at least starter and finisher)

    Returns:
        dict containing the following k-v pairs:
            "starter": dict with the starter state dict
            "intermediate": list containing the intermediate state dicts
            "finisher": dict with the finisher state dict
    """
    # TODO: make more efficient (too many nested loops) - maybe
    # Set up some parameters - they are used to gather the relevant keys
    base_name_transformer = "transformer"
    tok_emb = "token_embedding"
    pos_emb = "position_embedding"
    layer_name = "layers"
    transformer_last = f"{base_name_transformer}.ln_f"
    output_layer = "lm_head"
    n_intermediate_nodes = n_nodes - 2

    assert n_nodes >= 2

    out_chunks = {}

    # 1. Select params for Starter
    out_chunks["starter"] = {}
    out_chunks["starter"][f"starter_model.{tok_emb}.weight"] = model_params[
        f"{base_name_transformer}.{tok_emb}.weight"
    ]
    out_chunks["starter"][f"starter_model.{pos_emb}.weight"] = model_params[
        f"{base_name_transformer}.{pos_emb}.weight"
    ]
    if f"{base_name_transformer}.{tok_emb}.bias" in model_params.keys():
        out_chunks["starter"][f"starter_model.{tok_emb}.bias"] = model_params[
            f"{base_name_transformer}.{tok_emb}.bias"
        ]

    if f"{base_name_transformer}.{pos_emb}.bias" in model_params.keys():
        out_chunks["starter"][f"starter_model.{pos_emb}.bias"] = model_params[
            f"{base_name_transformer}.{pos_emb}.bias"
        ]

    # Starter may have transformer layers
    valid_layer_ind = list(range(0, N_LAYERS_START))
    relevant_keys = [
        k
        for k in list(model_params.keys())
        if (
            k.startswith(f"{base_name_transformer}.{layer_name}")
            and int(k.split(".")[2]) in valid_layer_ind
        )
    ]

    for loc_ind, layer_ind in enumerate(valid_layer_ind):
        prefix = f"{base_name_transformer}.{layer_name}.{layer_ind}"
        for k in relevant_keys:
            if k.startswith(prefix):
                end = remove_prefix(k, prefix)
                new_k = f"starter_model.layers.{loc_ind}{end}"
                out_chunks["starter"][new_k] = model_params[k]

    # 2. Select params for every Intermediate
    out_chunks["intermediate"] = []
    for i in range(1, n_nodes - 1):
        curr_params = {}

        # Complicated pythonic list call to select the correct keys to be
        # transferred to the intermediate node
        # As reference, the keys for the layers all start with:
        #       transformer.layer.<layer_ind>.[...]
        # so we need to select the correct layer indices
        valid_layer_ind = [
            N_LAYERS_START + n
            for n in list(range((i - 1) * N_LAYERS_INTERM, i * N_LAYERS_INTERM))
        ]
        relevant_keys = [
            k
            for k in list(model_params.keys())
            if (
                k.startswith(f"{base_name_transformer}.{layer_name}")
                and int(k.split(".")[2]) in valid_layer_ind
            )
        ]

        # Iterate over old keys, select correct, create new keys, copy val
        local_layer_ind = 0
        for ind in valid_layer_ind:
            prefix = f"{base_name_transformer}.{layer_name}.{ind}"
            for k in relevant_keys:
                if k.startswith(prefix):
                    end = remove_prefix(k, prefix)
                    new_k = f"intermediate_model.layers.{local_layer_ind}{end}"
                    curr_params[new_k] = model_params[k]
            local_layer_ind += 1

        out_chunks["intermediate"].append(curr_params)

    # 3. Select params for Finisher
    out_chunks["finisher"] = {}

    # Layers:
    valid_layer_ind = list(
        range((n_nodes - 2) * N_LAYERS_FINISH, (n_nodes - 1) * N_LAYERS_FINISH)
    )
    valid_layer_ind = [
        N_LAYERS_START + n_intermediate_nodes * N_LAYERS_INTERM + k
        for k in range(N_LAYERS_FINISH)
    ]
    relevant_keys = [
        k
        for k in list(model_params.keys())
        if (
            k.startswith(f"{base_name_transformer}.{layer_name}")
            and int(k.split(".")[2]) in valid_layer_ind
        )
    ]
    local_layer_ind = 0
    for ind in valid_layer_ind:
        prefix = f"{base_name_transformer}.{layer_name}.{ind}"
        for k in relevant_keys:
            if k.startswith(prefix):
                end = remove_prefix(k, prefix)
                new_k = f"finisher_model.layers.{local_layer_ind}{end}"
                out_chunks["finisher"][new_k] = model_params[k]
        local_layer_ind += 1
    # OK below
    out_chunks["finisher"][f"finisher_model.ln_f.weight"] = model_params[
        f"{transformer_last}.weight"
    ]
    try:
        out_chunks["finisher"][f"finisher_model.ln_f.bias"] = model_params[
            f"{transformer_last}.bias"
        ]
    except:
        pass

    out_chunks["finisher"][f"finisher_model.lm_head.weight"] = model_params[
        f"{output_layer}.weight"
    ]
    try:
        out_chunks["finisher"][f"finisher_model.lm_head.bias"] = model_params[
            f"{output_layer}.bias"
        ]
    except:
        pass

    return out_chunks


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
    De-serialize a dictionary and return a state dictionary containing a torch
    model parameters.
    """
    deserialized_params = {}
    for key, value in params.items():
        if isinstance(value, list):
            # Convert lists back to PyTorch tensors
            deserialized_params[key] = torch.tensor(value)
        else:
            deserialized_params[key] = value

    return deserialized_params
