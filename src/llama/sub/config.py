#!/usr/bin/env python3

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Self, Type, Union

import torch

from .utils import find_multiple

"""
Configurator
"""

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# DEVICE = "cpu"  # On macOS there are some issues with MPS

# ---- Training configuration --------------------
INIT_FROM = "scratch"  # "scratch", "resume" or "gpt2"

MAX_ITERS = N_ITER_TRAIN = 600000  # total number of training iterations
# Loss evaluation
CKPT_INTERVAL = 2000  # Iters between each ckpt update
EVAL_ITERS = 200
LOG_INTERVAL = 10
EVAL_ONLY = False  # if True, script exits right after the first eval
ALWAYS_SAVE_CHECKPOINT = False  # T: always save a checkpoint after each eval

# Optimizer settings (AdamW)
WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0  # clip gradients at this value, or disable if == 0.0

# Learning rate (& decay)
LEARNING_RATE: float = 3e-4
DECAY_LR: bool = True
WARMUP_ITERS: int = 2000
LR_DECAY_ITERS: int = 600000
MIN_LR: float = 6e-5  # ~= .1*lr

# ---- Generation settings ----------------------
TOP_K = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
TEMPERATURE = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)

# ---- MDI settings ------------------------------
# Adaptive layer number - first key is the number of nodes, second is the node type
N_LAYERS_NODES = {
    2: {
        5: {"N_LAYERS_START": 2, "N_LAYERS_SECONDARY": 3},
        7: {"N_LAYERS_START": 3, "N_LAYERS_SECONDARY": 4},
        9: {"N_LAYERS_START": 4, "N_LAYERS_SECONDARY": 5},
        12: {"N_LAYERS_START": 5, "N_LAYERS_SECONDARY": 7},  # gpt2
        24: {"N_LAYERS_START": 10, "N_LAYERS_SECONDARY": 14},  # gpt2-medium
        36: {"N_LAYERS_START": 16, "N_LAYERS_SECONDARY": 20},  # gpt2-large
        48: {"N_LAYERS_START": 22, "N_LAYERS_SECONDARY": 26},  # gpt2-xl
    },
    3: {
        5: {"N_LAYERS_START": 1, "N_LAYERS_SECONDARY": 2},
        7: {"N_LAYERS_START": 1, "N_LAYERS_SECONDARY": 3},
        9: {"N_LAYERS_START": 1, "N_LAYERS_SECONDARY": 4},
        12: {"N_LAYERS_START": 2, "N_LAYERS_SECONDARY": 5},  # gpt2
        24: {"N_LAYERS_START": 4, "N_LAYERS_SECONDARY": 10},  # gpt2-medium
        36: {"N_LAYERS_START": 10, "N_LAYERS_SECONDARY": 13},  # gpt2-large
        48: {"N_LAYERS_START": 14, "N_LAYERS_SECONDARY": 17},  # gpt2-xl
    },
}

HEADERLENGTH = 16  # Header length in chars
MSGLENGTH = 16 * 2048  # Message length in characters

# ---- System configuration ----------------------
DTYPE = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
COMPILE = False  # use PyTorch 2.0 to compile the model to be faster

# DDP settings
BACKEND = "nccl"  # 'nccl', 'gloo', etc.

# ---- Runtime configuration ---------------------
VERB = False
DEBUG = False
PLOTS = False

# ---- Configuration - as in LitGPT ---------------------------------------------------


@dataclass
class Config:
    """
    Model configuration.

    This class specifies all the parameters that characterize a specific model.

    Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file at
    https://github.com/Lightning-AI/litgpt/blob/main/LICENSE.
    """
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    scale_embeddings: bool = False
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    head_size: Optional[int] = None
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = (
        "GptNeoxMLP"
    )
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    n_expert: int = 0
    n_expert_per_token: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(
                    f"The config {self.name!r}, needs to set the `intermediate_size`"
                )
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        """
        Load model configuration given its name on Huggingface 🤗
        """
        if name not in name_to_config:
            # search through all `config['hf_config']['name']`
            try:
                conf_dict = next(
                    config for config in configs if name == config["hf_config"]["name"]
                )
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    # TODO: maybe remove? No benefit to MDI-LLM, but can be better to use YAML
    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    # FIXME: make it compliant
    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> Self:
        """Automatically load `model_config.yaml` and if it doesn't exist - a matching config from `litgpt/config.py`."""
        if (config_path := path / "model_config.yaml").is_file():
            return cls.from_file(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(
            f"For {str(path)!r} neither 'model_config.yaml' nor matching config exists."
        )

    # TODO: remove, as the MLP class used will be only one (LLaMa's)
    @property
    def mlp_class(self) -> Type:
        # `self.mlp_class_name` cannot be the type to keep the config serializable
        return getattr(litgpt.model, self.mlp_class_name)

    # TODO: remove, as the norm class used will be only one (LLaMa's)
    @property
    def norm_class(self) -> Type:
        # `self.norm_class_name` cannot be the type to keep the config serializable
        if self.norm_class_name == "RMSNorm":
            from functools import partial

            from model import RMSNorm

            return partial(RMSNorm, add_unit_offset="Gemma" in self.name)
        return getattr(torch.nn, self.norm_class_name)


configs: List[Dict] = []

##############################
# OpenLM Research Open LLaMA #
##############################
open_LLaMA = [
    # https://huggingface.co/openlm-research/open_llama_3b/blob/main/config.json
    dict(
        name="open_llama_3b",
        hf_config=dict(org="openlm-research", name="open_llama_3b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=26,
        n_embd=3200,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="LLaMAMLP",
        intermediate_size=8640,
    ),
    # https://huggingface.co/openlm-research/open_llama_7b/blob/main/config.json
    dict(
        name="open_llama_7b",
        hf_config=dict(org="openlm-research", name="open_llama_7b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/openlm-research/open_llama_13b/blob/main/config.json
    dict(
        name="open_llama_13b",
        hf_config=dict(org="openlm-research", name="open_llama_13b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
]
configs.extend(open_LLaMA)

#############
# TinyLLaMa #
#############
tiny_llama = [
    dict(
        name="tiny-llama-1.1b{}",
        hf_config=dict(org="TinyLlama", name="TinyLlama-1.1B{}"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=32,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",  # original TinyLlama uses FusedRMSNorm
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=4,
    )
]

for c in tiny_llama:
    for kind, hf_postfix in (
        ("", "-intermediate-step-1431k-3T"),
        ("-chat", "-Chat-v1.0"),
    ):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(hf_postfix)
        configs.append(copy)

################
# Meta LLaMa 2 #
################
llama_2 = [
    # https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
    dict(
        name="Llama-2-7b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-7b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
    dict(
        name="Llama-2-13b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-13b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
    dict(
        name="Llama-2-70b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-70b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

###############
# Meta LLaMA 3
###############
llama_3 = [
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
    dict(
        name="Llama-3-8B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3-8B{}"),
        block_size=8192,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
        rope_base=500000,
    ),
    # https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json
    dict(
        name="Llama-3-70B{}",
        hf_config=dict(org="meta-llama", name="Meta-Llama-3-70B{}"),
        block_size=8192,
        vocab_size=128000,
        padded_vocab_size=128256,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=500000,
    ),
]
for c in llama_3:
    for kind in ("", "-Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

name_to_config = {config["name"]: config for config in configs}
