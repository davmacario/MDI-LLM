#!/usr/bin/env python3

import gc
import math
import os
import sys
import threading
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import yaml
from numpy.typing import NDArray
from torch import nn

from sub.config import (EVAL_ITERS, LEARNING_RATE, LR_DECAY_ITERS, MIN_LR,
                        N_LAYERS_NODES, WARMUP_ITERS)
from sub.model import Config
from sub.utils.data_loader import get_batch
from sub.utils.typing import JSONType

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
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
                )
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


def waiting_animation(text: str, stopping: threading.Event):
    steps = ["⠴", "⠦", "⠇", "⠋", "⠙", "⠸"]
    stopping.clear()
    ind = 0
    while not stopping.is_set():
        print(text + f" {steps[ind]}", end="\r")
        ind += 1
        ind %= len(steps)
        time.sleep(0.5)
    print("")


def remove_prefix(text: str, prefix: str) -> str:
    """
    Remove the specified prefix from the given string.
    NOTE: starting Python 3.9, use text.removeprefix(prefix);
    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def find_eot(
    tokens: torch.Tensor,
    stop_tokens: Tuple[List[int], ...] = (),
    prompt_length: int = 0,
) -> torch.Tensor:
    """
    Return the sequence of tokens until the stopping tokens are found.
    The function finds the first EOS sequence starting from `prompt_length` (default 0)
    onwards.
    It will return the tensor truncated at the first EOS sequence after the prompt.

    Args:
        tokens: output of the LLM
        stop_tokens: tuple containing lists of the IDs representing the EOS
        prompt_length: optional prompt length
    """
    tok_lst = tokens.view(-1, 1).squeeze().tolist()
    assert (
        len(tok_lst) >= prompt_length
    ), "Prompt length must be longer than the provided tensor"
    start_ind = prompt_length + max([len(st) for st in stop_tokens])  # Skip prompt
    for i in range(start_ind, len(tok_lst)):
        if any(
            all(a == b for a, b in zip(tok_lst[i - len(st) : i], st))
            for st in stop_tokens
        ):
            return tokens[:, :i]
    return tokens


def detect_stop_tokens(
    tokens: torch.Tensor, stop_tokens: Tuple[List[int], ...] = ()
) -> bool:
    """
    Will return True if `tokens` terminates with one of the sequences defined in
    `stop_tokens`.
    """
    tok_lst = tokens.view(-1, 1).squeeze().tolist()
    return any(
        all(a == b for a, b in zip(tok_lst[-len(st) :], st)) for st in stop_tokens
    )


def format_output(text: str):
    """
    Display the generated text correctly;

    This requires to isolate the <|user|> and <|assistant|> elements to isolate the
    specific things said by each.


    Maybe format with color??
    """
    pass


def split_parameters(
    model_params: Dict[str, Any], n_nodes: int
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Split the model parameters (contained in a state dict) among the different
    available nodes.
    The model structure is that of LitGPT (https://github.com/Lightning-AI/litgpt).

    The number of nodes should be at least 2 (starter and finisher).

    The parameters are divided as such:
        - Starter: token embedding,
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
    base_name_starter = "transformer"
    base_name_secondary = "transformer"
    tok_emb = "wte"
    layer_name = "h"
    transformer_last = f"{base_name_transformer}.ln_f"
    output_layer = "lm_head"  # outside transformer now

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
    out_chunks["starter"][f"{base_name_starter}.{tok_emb}.weight"] = model_params.pop(
        f"{base_name_transformer}.{tok_emb}.weight"
    )
    if f"{base_name_transformer}.{tok_emb}.bias" in model_params.keys():
        out_chunks["starter"][f"{base_name_starter}.{tok_emb}.bias"] = model_params.pop(
            f"{base_name_transformer}.{tok_emb}.bias"
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
        new_k = f"{base_name_starter}.{layer_name}.{ind_layer_chunk}.{end}"
        out_chunks["starter"][new_k] = model_params.pop(k_orig)

    # ln_f - last layernorm
    out_chunks["starter"][f"{base_name_starter}.ln_f.weight"] = model_params.pop(
        f"{transformer_last}.weight"
    )
    if f"{transformer_last}.bias" in model_params.keys():
        out_chunks["starter"][f"{base_name_starter}.ln_f.bias"] = model_params.pop(
            f"{transformer_last}.bias"
        )

    # lm_head - final linear layers (not in 'transformer')
    out_chunks["starter"][f"lm_head.weight"] = model_params.pop(
        f"{output_layer}.weight"
    )
    if f"{output_layer}.bias" in model_params.keys():
        out_chunks["starter"][f"lm_head.bias"] = model_params.pop(
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
            new_k = f"{base_name_secondary}.{layer_name}.{ind_layer_chunk}.{end}"
            curr_params[new_k] = model_params.pop(k_orig)

        out_chunks["secondary"].append(curr_params)

    return out_chunks, layers_info


def split_and_store(
    model_params: Dict[str, Any],
    n_nodes: int,
    ckpt_dir: Union[Path, str],
    **kwargs,
) -> Path:
    """
    Given a state dict, split it among a number of nodes following the configuration.

    Args:
        model_params: state dict
        n_nodes: number of nodes among which to split the model
        ckpt_dir: checkpoint directory of the model

    Returns:
        path of the chunks subdirectory (ckpt_dir/chunks/<n>nodes/)
    """
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)

    verb = False if "verb" not in kwargs else kwargs["verb"]

    chunks, layer_info = split_parameters(model_params, n_nodes)
    if len(model_params):
        warnings.warn(f"{len(model_params)} elements have not been used")
    del model_params
    gc.collect()

    n_secondary = n_nodes - 1

    if verb:
        print("Using the following split:")
        print(f"- Starter node: {layer_info['N_LAYERS_START']} layers")
        print(
            f"- {n_secondary} secondary node{'s' if n_secondary - 1 else ''}: "
            f"{layer_info['N_LAYERS_SECONDARY']} layers"
        )

    chunks_subfolder = ckpt_dir / "chunks" / f"{n_nodes}nodes"
    os.makedirs(chunks_subfolder, exist_ok=True)

    # Starter
    starter_file = chunks_subfolder / "model_starter.pth"
    torch.save(chunks["starter"], starter_file)

    # Secondary (NOTE: zero-indexing in file name)
    for i in range(n_secondary):
        current_file = chunks_subfolder / f"model_secondary{i}.pth"
        torch.save(chunks["secondary"][i], current_file)

    return chunks_subfolder


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


def count_transformer_blocks(
    state_dict: Dict[str, Any], base_name_transformer: Optional[str] = "transformer"
) -> int:
    """
    Given a state dict, return the number of detected transformer blocks.
    The default name for the transformer blocks is `transformer`, but can be overridden
    to support a different naming convention.

    Args:
        state_dict: dict containing the model parameters.
        base_name_transformer: base name of the transformer block, i.e., first "key" in
            the dict.
    """
    layer_name = "h"

    # Count the number of detected transformer layers
    layer_keys = [
        k
        for k in state_dict.keys()
        if k.startswith(f"{base_name_transformer}.{layer_name}")
    ]
    layers_unique = list(set([".".join(k.split(".")[:3]) for k in layer_keys]))
    return len(layers_unique)


def load_sd(
    model_path: Path, device: Optional[Union[torch.device, str]] = "cpu"
) -> Dict[str, Any]:
    """
    Load a state dictionary (model parameters).

    Args:
        model_path: path of the file (typically .pt or .pth) containing the model
            parameters.
        device (default "cpu"): device where to load the weights (NOT the model!)

    Returns:
        state dict of the model (can be passed to a compatible nn.Module object through
            the method `nn.Module.load_state_dict()`.
    """
    try:
        sd = torch.load(model_path, map_location=device)
    except Exception as e:
        if "out of memory" in str(e):
            if device != "cpu":
                warnings.warn(
                    f"Unable to fit model ckpt in {device} memory! Retrying with cpu"
                )
                sd = torch.load(model_path, map_location="cpu")
            else:
                raise e
        else:
            raise e

    return sd


def load_from_pt(
    model_path: Union[Path, str],
    device: Optional[Union[torch.device, str]] = "cpu",
    config_only: Optional[bool] = False,
) -> Tuple[Config, Optional[Dict[str, Any]]]:
    """
    Load model weights from disk.

    Args:
        model_path: path to the checkpoint
        device (default: "cpu"): device where to load state dict; default: "cpu"
        config_only (default: False): if True, only return the Config object

    Returns:
        model config (Config object)
        [model state dictionary, compatible with GPT class]
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

    if config_only:
        return config, None

    pth_file = model_dir / "lit_model.pth"
    sd = load_sd(pth_file, device)

    return config, sd


def load_from_hf(
    repo_id: str,
    access_token: Optional[str] = None,
    dtype: Optional[str] = None,
    checkpoint_dir: Path = Path("checkpoints"),
    model_name: Optional[str] = None,
    device: Optional[str] = "cpu",
    config_only: Optional[bool] = False,
) -> Tuple[Config, Optional[Dict[str, Any]]]:
    """
    Load model weights from Huggingface.
    It saves the files to the checkpoint directory, converts them to the right format
    and loads the model configuration and the state dict.

    Args:
        repo_id: Huggingface Hub repository ID
        access_token: optional API token for accessing private Huggingface models
        dtype: data type for the downloaded weights
        checkpoint_dir: path of the directory where to place the model folders
        model_name: the existing config name to use for this `repo_id`. This is
            useful to download alternative weights of existing architectures.
        device: device where to load state dict
        config_only (default: False): if true, only return the Config object (note that
            the model will be downloaded anyways)

    Returns:
        model config (Config object)
        [model state dictionary, compatible with GPT class]
    """
    from .download import download_from_hub

    download_from_hub(
        repo_id=repo_id,
        access_token=access_token,
        dtype=dtype,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
    )

    model_path = checkpoint_dir / repo_id
    return load_from_pt(model_path, device, config_only=config_only)


def get_available_models(ckpt_dir: Path) -> List[JSONType]:
    """
    Provides a list of dicts containing the information of the models for which at least
    one chunk has been found in the local node.

    Will return, for each of the subdirectories at 'ckpt_dir', a dict:
    {
      "name": model name
      "hf_config": {
        "org": organization name (e.g., mistralai),
        "name": actual model name
      },
      "chunks": {
        "<n>nodes": [...],   <-- List of all the chunks (list dir)
        "<k>nodes": [...]
      }
    }
    """
    # No recursion because (believe it or not) it's messier
    out = []
    for org in ckpt_dir.iterdir():
        if org.is_dir():
            curr_org_name = org.name
            for mod in org.iterdir():
                new_mod_dict = {}
                new_mod_dict["name"] = mod.name
                new_mod_dict["hf_config"] = dict(org=curr_org_name, name=mod.name)

                curr_chunk_dir = mod / "chunks"
                if curr_chunk_dir.is_dir():
                    new_mod_dict["chunks"] = {}
                    # First, look for chunks, then add info only if chunks have been found
                    tmp_mod_dict = {}
                    for p in curr_chunk_dir.rglob("*"):
                        if not p.is_dir():
                            # We have a chunk
                            if p.parent.name not in tmp_mod_dict:
                                # Create list
                                tmp_mod_dict["p.parent.name"] = []
                            # Append chunk name
                            tmp_mod_dict["p.parent.name"].append(p.name)

                    # Only add keys with non-empty lists
                    for k, v in tmp_mod_dict.items():
                        if len(v):
                            new_mod_dict["chunks"][k] = v
                    
                    # Only add the model info if some chunks have been found
                    if len(new_mod_dict["chunks"]):
                        out.append(new_mod_dict)

    return out


def save_config(config: "Config", checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)


def s_to_ns(timestamp_s):
    return int(timestamp_s * 1e9)
