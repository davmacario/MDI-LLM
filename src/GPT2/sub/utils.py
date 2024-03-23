#!/usr/bin/env python3

import gc
import math
import os
import sys
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Tuple, Union

import matplotlib.pyplot as plt
import torch
from numpy.typing import NDArray
from torch import nn
from transformers import GPT2LMHeadModel

from .config import (DEVICE, EVAL_ITERS, LEARNING_RATE, LR_DECAY_ITERS, MIN_LR,
                     N_LAYERS_NODES, VERB, WARMUP_ITERS)
from .data_loader import get_batch
from .model import GPT, GPTConfig


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
    model: Union[GPT, nn.Module],
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


def find_eot(text: str) -> int:
    """
    Return the index of the first character of '<|endoftext|>', if found in text.
    Else, return len(text)
    """
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
            N_LAYERS_STARTxTransformer Layers + final linear layer
        - Intermediate: N_LAYERS_INTERMxTransformer Layer
        - Finisher: N_LAYERS_FINISHxTransformer Layer, LayerNorm

    Args:
        model_params: complete model parameters (state dict)
        n_nodes: number of nodes among which to divide the parameters; must be
        greater or equal to 2 (at least starter and finisher)

    Returns:
        dict containing the following k-v pairs:
            "starter": dict with the starter state dict
            "intermediate": list containing the intermediate state dicts
            "finisher": dict with the finisher state dict
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
    n_intermediate_nodes = n_nodes - 2

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
    if n_nodes > 2:
        n_layers_interm = N_LAYERS_NODES[n_nodes][n_layers_model]["N_LAYERS_INTERM"]
        layers_info["N_LAYERS_INTERM"] = n_layers_interm
    else:
        n_layers_interm = 0
        layers_info["N_LAYERS_INTERM"] = n_layers_interm

    n_layers_finish = N_LAYERS_NODES[n_nodes][n_layers_model]["N_LAYERS_FINISH"]
    layers_info["N_LAYERS_FINISH"] = n_layers_finish

    if VERB:
        print(f"Number of layers - starter node: {n_layers_start}")
        if n_nodes > 2:
            print(f"Number of layers - intermediate node: {n_layers_interm}")
        print(f"Number of layers - finisher node: {n_layers_finish}")

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

    # Starter may have transformer layers
    valid_layer_ind = list(range(0, n_layers_start))
    relevant_keys = [
        k
        for k in list(model_params.keys())
        if (
            k.startswith(f"{base_name_transformer}.{layer_name}")
            and int(k.split(".")[2]) in valid_layer_ind
        )
    ]

    for loc_ind, layer_ind in enumerate(valid_layer_ind):
        prefix = f"{base_name_transformer}.{layer_name}.{layer_ind}."
        for k in relevant_keys:
            if k.startswith(prefix):
                end = remove_prefix(k, prefix)
                new_k = f"starter_model.{layer_name}.{loc_ind}{end}"
                out_chunks["starter"][new_k] = model_params.pop(k)

    out_chunks["starter"][f"starter_model.lm_head.weight"] = model_params.pop(
        f"{output_layer}.weight"
    )
    if f"{output_layer}.bias" in model_params.keys():
        out_chunks["starter"][f"starter_model.lm_head.bias"] = model_params.pop(
            f"{output_layer}.bias"
        )

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
            n_layers_start + n
            for n in list(range((i - 1) * n_layers_interm, i * n_layers_interm))
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
            prefix = f"{base_name_transformer}.{layer_name}.{ind}."
            for k in relevant_keys:
                if k.startswith(prefix):
                    end = remove_prefix(k, prefix)
                    new_k = f"intermediate_model.{layer_name}.{local_layer_ind}{end}"
                    curr_params[new_k] = model_params.pop(k)
            local_layer_ind += 1

        out_chunks["intermediate"].append(curr_params)

    # 3. Select params for Finisher
    out_chunks["finisher"] = {}

    # Layers:
    valid_layer_ind = list(
        range((n_nodes - 2) * n_layers_finish, (n_nodes - 1) * n_layers_finish)
    )
    valid_layer_ind = [
        n_layers_start + n_intermediate_nodes * n_layers_interm + k
        for k in range(n_layers_finish)
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
        prefix = f"{base_name_transformer}.{layer_name}.{ind}."
        for k in relevant_keys:
            if k.startswith(prefix):
                end = remove_prefix(k, prefix)
                new_k = f"finisher_model.{layer_name}.{local_layer_ind}{end}"
                out_chunks["finisher"][new_k] = model_params.pop(k)
        local_layer_ind += 1

    out_chunks["finisher"][f"finisher_model.ln_f.weight"] = model_params.pop(
        f"{transformer_last}.weight"
    )
    if f"{transformer_last}.bias" in model_params.keys():
        out_chunks["finisher"][f"finisher_model.ln_f.bias"] = model_params.pop(
            f"{transformer_last}.bias"
        )

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


def count_model_layers(model_params: Dict[str, Any]) -> int:
    base_name_transformer = "transformer"
    layer_name = "h"

    # Count the number of detected transformer layers
    layer_keys = [
        k
        for k in model_params.keys()
        if k.startswith(f"{base_name_transformer}.{layer_name}")
    ]
    layers_unique = list(set([".".join(k.split(".")[:3]) for k in layer_keys]))
    return len(layers_unique)


def load_from_pt(model_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load model weights from disk.

    Args:
        model_path: path to the checkpoint

    Returns:
        model state dictionary
        model config (args of GPTConfig) as dictionary
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Unable to find model checkpoint at {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
    except:
        if DEVICE != "cpu":
            warnings.warn(
                f"Unable to fit model ckpt in {DEVICE} memory! Retrying with cpu"
            )
            checkpoint = torch.load(model_path, map_location="cpu")
        else:
            raise MemoryError("Not enough system memory to load ckpt!")

    return checkpoint["model"], checkpoint["model_args"]


def load_from_hf(model_type: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load model weights from Huggingface.

    Args:
        model_type: one of ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

    Returns:
        model state dictionary, imported from gpt2
        model config (arguments of GPTConfig) as dictionary
    """
    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

    config_args = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }[model_type]

    config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
    config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
    config_args["bias"] = True  # always True for GPT model checkpoints
    if "dropout" not in config_args:
        config_args[
            "dropout"
        ] = 0.0  # NOTE: this assumes chunk won't be used for training

    # TODO: remove creation of GPT object (too much memory) - pay attention to .attn.bias
    # attn.bias is just the triangular masking matrix used to compute attention
    # Should not be a problem to use default one (?) - since we are not using flash attn
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    sd_keys_hf = [
        k
        for k in sd_hf.keys()
        if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))
    ]

    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]
    assert len(sd_keys_hf) == len(
        sd_keys
    ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # Special treatment for Conv1D parameters
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # Vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return sd, config_args


# ---------- PLOTS -------------------------------------------------------------
file_dir = os.path.dirname(__file__)


def plot_tokens_per_time(
    tok_time: List[Union[Tuple, List[Tuple]]],
    out_path: str = os.path.join(file_dir, "..", "img", "tokens_time.png"),
    disp: bool = True,
):
    """
    Plot a graph representing the number of generated tokens in time.

    Args:
        tok_time: list of couples, where the 1st element is the number of
            samples and the 2nd element is the time at which it was generated.
            It can also be a list of list of couples (multiple samples); in this
            case, the plot will distinguish between the different samples
        out_path: path of the produced output image
        disp: if true, the image will also be displayed at runtime
    """
    assert len(tok_time) >= 1

    fig = plt.figure(figsize=(12, 8))
    if isinstance(tok_time[0], Tuple):
        time = [x[1] for x in tok_time]
        n_samples = [x[0] for x in tok_time]
        plt.plot(time, n_samples)
        plt.title("Number of generated samples vs. time - MDI")
    elif isinstance(tok_time[0], List):
        for i, sublist in enumerate(tok_time):
            time = [x[1] for x in sublist]
            n_samples = [x[0] for x in sublist]
            plt.plot(time, n_samples, label=f"Sample {i + 1}")
            plt.legend()
        plt.title("Number of generated samples vs. time - standalone")
    plt.xlabel("Time (s)")
    plt.ylabel("N. samples")
    plt.grid()
    plt.tight_layout()
    fig.savefig(out_path)
    if disp:
        plt.show()
