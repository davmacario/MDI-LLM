#!/usr/bin/env python3

from copy import deepcopy
from typing import Dict, List

import torch

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
BATCH_SIZE = 24
MAX_ITERS = N_ITER_TRAIN = 600000  # total number of training iterations
GRADIENT_ACCUMULATION_STEPS = 4
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
        22: {"N_LAYERS_START": 10, "N_LAYERS_SECONDARY": 12},  # TinyLlama
        24: {"N_LAYERS_START": 10, "N_LAYERS_SECONDARY": 14},  # gpt2-medium
        32: {"N_LAYERS_START": 14, "N_LAYERS_SECONDARY": 18},  # Llama 2
        36: {"N_LAYERS_START": 16, "N_LAYERS_SECONDARY": 20},  # gpt2-large
        48: {"N_LAYERS_START": 22, "N_LAYERS_SECONDARY": 26},  # gpt2-xl
    },
    3: {
        5: {"N_LAYERS_START": 1, "N_LAYERS_SECONDARY": 2},
        7: {"N_LAYERS_START": 1, "N_LAYERS_SECONDARY": 3},
        9: {"N_LAYERS_START": 1, "N_LAYERS_SECONDARY": 4},
        12: {"N_LAYERS_START": 2, "N_LAYERS_SECONDARY": 5},  # gpt2
        22: {"N_LAYERS_START": 6, "N_LAYERS_SECONDARY": 8},  # TinyLlama
        24: {"N_LAYERS_START": 4, "N_LAYERS_SECONDARY": 10},  # gpt2-medium
        32: {"N_LAYERS_START": 8, "N_LAYERS_SECONDARY": 12},  # Llama 2
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
DTYPE_TORCH = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[DTYPE]
COMPILE = False  # use PyTorch 2.0 to compile the model to be faster

# DDP settings
BACKEND = "nccl"  # 'nccl', 'gloo', etc.

# ---- Runtime configuration ---------------------
VERB = False
DEBUG = False
PLOTS = False

# ---- Configuration - as in LitGPT ---------------------------------------------------

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
