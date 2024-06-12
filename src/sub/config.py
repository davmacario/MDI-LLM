#!/usr/bin/env python3

from copy import deepcopy
from typing import Any, Dict, List

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
    1: {
        5: {"N_LAYERS_START": 5},
        7: {"N_LAYERS_START": 7},
        9: {"N_LAYERS_START": 9},
        12: {"N_LAYERS_START": 12},  # gpt2
        22: {"N_LAYERS_START": 22},  # TinyLlama
        24: {"N_LAYERS_START": 24},  # gpt2-medium
        32: {"N_LAYERS_START": 32},  # Llama 2
        36: {"N_LAYERS_START": 36},  # gpt2-large
        48: {"N_LAYERS_START": 48},  # gpt2-xl
    },
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
DTYPE_TORCH_MAPPING = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
DTYPE_TORCH = DTYPE_TORCH_MAPPING[DTYPE]
COMPILE = False  # use PyTorch 2.0 to compile the model to be faster


class TrainingConfig:
    tie_embeddings: bool = True
    learning_rate = LEARNING_RATE
    decay_lr = DECAY_LR
    min_lr = MIN_LR
    warmup_iters = WARMUP_ITERS
    weight_decay = WEIGHT_DECAY
    beta1 = BETA1
    beta2 = BETA2
    grad_clip = GRAD_CLIP
    # Modifiable
    compile = COMPILE
    _dtype_torch = DTYPE_TORCH
    eval_only = EVAL_ONLY
    batch_size = BATCH_SIZE
    max_iters = MAX_ITERS
    ckpt_interval = CKPT_INTERVAL
    log_interval = LOG_INTERVAL
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    device = DEVICE
    compile = COMPILE

    def as_dict(self) -> Dict[str, Any]:
        out = {}
        for attr in dir(self):
            if not attr.startswith("__") and not attr.endswith("__"):
                out[attr] = getattr(self, attr, None)
                if out[attr] is None:
                    del out[attr]
        return out

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype_torch

    @dtype.setter
    def dtype(self, value: str) -> None:
        """
        Automatically map a str dtype (among 'float32', 'bfloat16' and 'float16') to the
        correct torch.dtype
        """
        if value not in DTYPE_TORCH_MAPPING:
            raise ValueError(f"Supported dtypes are: {DTYPE_TORCH_MAPPING.keys()}")
        self._dtype_torch = DTYPE_TORCH_MAPPING[value]


# DDP settings
BACKEND = "nccl"  # 'nccl', 'gloo', etc.

# ---- Runtime configuration ---------------------
VERB = False
DEBUG = False
PLOTS = False

# ---- Configuration - as in LitGPT ---------------------------------------------------

configs: List[Dict] = []

########################
# Stability AI StableLM
########################
configs = [
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json
    dict(
        name="stablelm-base-alpha-3b",
        hf_config=dict(org="stabilityai", name="stablelm-base-alpha-3b"),
    ),
    # https://huggingface.co/stabilityai/stablelm-base-alpha-7b/blob/main/config.json
    dict(
        name="stablelm-base-alpha-7b",
        hf_config=dict(org="stabilityai", name="stablelm-base-alpha-7b"),
        n_head=48,
        n_embd=6144,
        padding_multiple=256,
    ),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/blob/main/config.json
    dict(
        name="stablelm-tuned-alpha-3b",
        hf_config=dict(org="stabilityai", name="stablelm-tuned-alpha-3b"),
        n_head=32,
    ),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b/blob/main/config.json
    dict(
        name="stablelm-tuned-alpha-7b",
        hf_config=dict(org="stabilityai", name="stablelm-tuned-alpha-7b"),
        n_head=48,
        n_embd=6144,
        padding_multiple=256,
    ),
    # https://huggingface.co/stabilityai/stablelm-3b-4e1t/blob/main/config.json
    dict(
        name="stablelm-3b-4e1t",
        hf_config=dict(org="stabilityai", name="stablelm-3b-4e1t"),
        padded_vocab_size=50304,
        n_layer=32,
        n_head=32,
        n_embd=2560,
        parallel_residual=False,
        bias=False,
        mlp_class_name="LLaMAMLP",
        intermediate_size=6912,
    ),
    # https://huggingface.co/stabilityai/stablelm-zephyr-3b/blob/main/config.json
    dict(
        name="stablelm-zephyr-3b",
        hf_config=dict(org="stabilityai", name="stablelm-zephyr-3b"),
        padded_vocab_size=50304,
        n_layer=32,
        n_head=32,
        n_embd=2560,
        parallel_residual=False,
        bias=False,
        mlp_class_name="LLaMAMLP",
        intermediate_size=6912,
    ),
]


##########################
# Stability AI StableCode
##########################
stablecode = [
    # https://huggingface.co/stabilityai/stablecode-completion-alpha-3b/blob/main/config.json
    dict(
        name="stablecode-completion-alpha-3b",
        hf_config=dict(org="stabilityai", name="stablecode-completion-alpha-3b"),
        block_size=16384,
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stablecode-completion-alpha-3b-4k/blob/main/config.json
    dict(
        name="stablecode-completion-alpha-3b-4k",
        hf_config=dict(org="stabilityai", name="stablecode-completion-alpha-3b-4k"),
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b/blob/main/config.json
    dict(
        name="stablecode-instruct-alpha-3b",
        hf_config=dict(org="stabilityai", name="stablecode-instruct-alpha-3b"),
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stable-code-3b/blob/main/config.json
    dict(
        name="stable-code-3b",
        hf_config=dict(org="stabilityai", name="stable-code-3b"),
        padded_vocab_size=50304,
        n_layer=32,
        n_embd=2560,
        block_size=16384,
        parallel_residual=False,
        bias=False,
        mlp_class_name="LLaMAMLP",
        intermediate_size=6912,
    ),
]
configs.extend(stablecode)


####################
# EleutherAI Pythia
####################
pythia = [
    # https://huggingface.co/EleutherAI/pythia-14m/blob/main/config.json
    dict(
        name="pythia-14m",
        hf_config=dict(org="EleutherAI", name="pythia-14m"),
        block_size=512,
        n_layer=6,
        n_embd=128,
        n_head=4,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-31m/blob/main/config.json
    dict(
        name="pythia-31m",
        hf_config=dict(org="EleutherAI", name="pythia-31m"),
        block_size=1024,
        n_layer=6,
        n_embd=256,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    dict(
        name="pythia-70m",
        hf_config=dict(org="EleutherAI", name="pythia-70m"),
        block_size=2048,
        n_layer=6,
        n_embd=512,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    dict(
        name="pythia-160m",
        hf_config=dict(org="EleutherAI", name="pythia-160m"),
        block_size=2048,
        n_layer=12,
        n_embd=768,
        n_head=12,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    dict(
        name="pythia-410m",
        hf_config=dict(org="EleutherAI", name="pythia-410m"),
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    dict(
        name="pythia-1b",
        hf_config=dict(org="EleutherAI", name="pythia-1b"),
        block_size=2048,
        n_embd=2048,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    dict(
        name="pythia-1.4b",
        hf_config=dict(org="EleutherAI", name="pythia-1.4b"),
        block_size=2048,
        n_layer=24,
        n_embd=2048,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    dict(
        name="pythia-2.8b",
        hf_config=dict(org="EleutherAI", name="pythia-2.8b"),
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-6.9b/blob/main/config.json
    dict(
        name="pythia-6.9b",
        hf_config=dict(org="EleutherAI", name="pythia-6.9b"),
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
    ),
    # https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json
    dict(
        name="pythia-12b",
        hf_config=dict(org="EleutherAI", name="pythia-12b"),
        block_size=2048,
        n_layer=36,
        n_embd=5120,
        n_head=40,
    ),
]
configs.extend(pythia)
for c in pythia:
    # "pythia-14m" and "pythia-31m" don't have deduped version
    if c["name"] in ("pythia-14m", "pythia-31m"):
        continue
    copy = deepcopy(c)
    copy["name"] = f"{c['name']}-deduped"
    copy["hf_config"]["name"] = f"{c['hf_config']['name']}-deduped"
    configs.append(copy)


###################
# databricks Dolly
###################
dolly = [
    # https://huggingface.co/databricks/dolly-v2-3b/blob/main/config.json
    dict(
        name="dolly-v2-3b",
        hf_config=dict(org="databricks", name="dolly-v2-3b"),
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padded_vocab_size=50280,
    ),
    # https://huggingface.co/databricks/dolly-v2-7b/blob/main/config.json
    dict(
        name="dolly-v2-7b",
        hf_config=dict(org="databricks", name="dolly-v2-7b"),
        block_size=2048,
        n_layer=32,
        padded_vocab_size=50280,
    ),
    # https://huggingface.co/databricks/dolly-v2-12b/blob/main/config.json
    dict(
        name="dolly-v2-12b",
        hf_config=dict(org="databricks", name="dolly-v2-12b"),
        block_size=2048,
        n_layer=36,
        n_embd=5120,
        n_head=40,
        padded_vocab_size=50280,
    ),
]
configs.extend(dolly)


####################################
# togethercomputer RedPajama INCITE
####################################
redpajama_incite = [
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1/blob/main/config.json
    dict(
        name="RedPajama-INCITE-{}-3B-v1",
        hf_config=dict(org="togethercomputer", name="RedPajama-INCITE-{}-3B-v1"),
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base/blob/main/config.json
    dict(
        name="RedPajama-INCITE-7B-{}",
        hf_config=dict(org="togethercomputer", name="RedPajama-INCITE-7B-{}"),
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # this redirects to the checkpoint above. kept for those who had the old weights already downloaded
    dict(
        name="RedPajama-INCITE-{}-7B-v0.1",
        hf_config=dict(org="togethercomputer", name="RedPajama-INCITE-{}-7B-v0.1"),
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
]
for c in redpajama_incite:
    for kind in ("Base", "Chat", "Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


#################
# TII UAE Falcon
#################
falcon = [
    # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
    dict(
        name="falcon-7b{}",
        hf_config=dict(org="tiiuae", name="falcon-7b{}"),
        block_size=2048,
        vocab_size=65024,
        padded_vocab_size=65024,
        n_layer=32,
        n_head=71,
        n_embd=4544,
        rotary_percentage=1.0,
        n_query_groups=1,
        bias=False,
        # this is not in the config, but in the original model implementation, only for this config
        shared_attention_norm=True,
    ),
    # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
    dict(
        name="falcon-40b{}",
        hf_config=dict(org="tiiuae", name="falcon-40b{}"),
        block_size=2048,
        vocab_size=65024,
        padded_vocab_size=65024,
        n_layer=60,
        n_head=128,
        n_embd=8192,
        rotary_percentage=1.0,
        n_query_groups=8,
        bias=False,
    ),
]
for c in falcon:
    for kind in ("", "-instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

# https://huggingface.co/tiiuae/falcon-180b/blob/main/config.json
falcon180b = dict(
    name="falcon-180B{}",
    hf_config=dict(org="tiiuae", name="falcon-180B{}"),
    block_size=2048,
    vocab_size=65024,
    padded_vocab_size=65024,
    n_layer=80,
    n_head=232,
    n_embd=14848,
    rotary_percentage=1.0,
    n_query_groups=8,
    bias=False,
)

for kind in ("", "-chat"):
    copy = deepcopy(falcon180b)
    copy["name"] = falcon180b["name"].format(kind)
    copy["hf_config"]["name"] = falcon180b["hf_config"]["name"].format(kind)
    configs.append(copy)


#############################
# OpenLM Research Open LLaMA
#############################
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


###############
# LMSYS Vicuna
###############
vicuna = [
    # https://huggingface.co/lmsys/vicuna-7b-v1.3/blob/main/config.json
    dict(
        name="vicuna-7b-v1.3",
        hf_config=dict(org="lmsys", name="vicuna-7b-v1.3"),
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
    # https://huggingface.co/lmsys/vicuna-13b-v1.3/blob/main/config.json
    dict(
        name="vicuna-13b-v1.3",
        hf_config=dict(org="lmsys", name="vicuna-13b-v1.3"),
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
    # https://huggingface.co/lmsys/vicuna-33b-v1.3/blob/main/config.json
    dict(
        name="vicuna-33b-v1.3",
        hf_config=dict(org="lmsys", name="vicuna-33b-v1.3"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=60,
        n_head=52,
        n_embd=6656,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="LLaMAMLP",
        intermediate_size=17920,
    ),
    # https://huggingface.co/lmsys/vicuna-7b-v1.5/blob/main/config.json
    dict(
        name="vicuna-7b-v1.5",
        hf_config=dict(org="lmsys", name="vicuna-7b-v1.5"),
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
    # https://huggingface.co/lmsys/vicuna-7b-v1.5-16k/blob/main/config.json
    dict(
        name="vicuna-7b-v1.5-16k",
        hf_config=dict(org="lmsys", name="vicuna-7b-v1.5-16k"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=4,
    ),
    # https://huggingface.co/lmsys/vicuna-13b-v1.5/blob/main/config.json
    dict(
        name="vicuna-13b-v1.5",
        hf_config=dict(org="lmsys", name="vicuna-13b-v1.5"),
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
    # https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/blob/main/config.json
    dict(
        name="vicuna-13b-v1.5-16k",
        hf_config=dict(org="lmsys", name="vicuna-13b-v1.5-16k"),
        block_size=16384,
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
        rope_condense_ratio=4,
    ),
]
configs.extend(vicuna)


#################
# LMSYS LongChat
#################
long_chat = [
    # https://huggingface.co/lmsys/longchat-7b-16k/blob/main/config.json
    dict(
        name="longchat-7b-16k",
        hf_config=dict(org="lmsys", name="longchat-7b-16k"),
        block_size=16384,
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
        rope_condense_ratio=8,
    ),
    # https://huggingface.co/lmsys/longchat-13b-16k/blob/main/config.json
    dict(
        name="longchat-13b-16k",
        hf_config=dict(org="lmsys", name="longchat-13b-16k"),
        block_size=16384,
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
        rope_condense_ratio=8,
    ),
]
configs.extend(long_chat)


######################
# NousResearch Hermes
######################
nous_research = [
    # https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b/blob/main/config.json
    dict(
        name="Nous-Hermes-llama-2-7b",
        hf_config=dict(org="NousResearch", name="Nous-Hermes-llama-2-7b"),
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/NousResearch/Nous-Hermes-13B/blob/main/config.json
    dict(
        name="Nous-Hermes-13b",
        hf_config=dict(org="NousResearch", name="Nous-Hermes-13b"),
        block_size=2048,
        vocab_size=32000,
        padded_vocab_size=32001,
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
    # https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b
    dict(
        name="Nous-Hermes-Llama2-13b",
        hf_config=dict(org="NousResearch", name="Nous-Hermes-Llama2-13b"),
        vocab_size=32000,
        padded_vocab_size=32032,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
]
configs.extend(nous_research)


###############
# Meta LLaMA 2
###############
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


###############
# Google Gemma
###############
gemma = [
    # https://huggingface.co/google/gemma-2b/blob/main/config.json
    dict(
        name="Gemma-2b",
        hf_config=dict(org="google", name="gemma-2b"),
        scale_embeddings=True,
        vocab_size=256000,
        padding_multiple=64,
        n_embd=2048,
        n_layer=18,
        n_head=8,
        n_query_groups=1,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        intermediate_size=16384,
    ),
    # https://huggingface.co/google/gemma-7b/blob/main/config.json
    dict(
        name="Gemma-7b",
        hf_config=dict(org="google", name="gemma-7b"),
        scale_embeddings=True,
        vocab_size=256000,
        padding_multiple=64,
        n_embd=3072,
        n_layer=28,
        n_head=16,
        head_size=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        intermediate_size=24576,
    ),
]
configs.extend(gemma)
for c in gemma:
    copy = deepcopy(c)
    copy["name"] = f"{c['name']}-it"
    copy["hf_config"]["name"] = f"{c['hf_config']['name']}-it"
    configs.append(copy)

##################
# Google CodeGemma
##################
codegemma = [
    # https://huggingface.co/google/codegemma-7b-it/blob/main/config.json
    dict(
        name="CodeGemma-7b-it",
        hf_config=dict(org="google", name="codegemma-7b-it"),
        scale_embeddings=True,
        vocab_size=256000,
        padding_multiple=64,
        n_embd=3072,
        n_layer=28,
        n_head=16,
        head_size=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GemmaMLP",
        gelu_approximate="tanh",
        intermediate_size=24576,
    ),
]
configs.extend(codegemma)

################
# H2Oai Danube2
################
danube2 = [
    # https://huggingface.co/h2oai/h2o-danube2-1.8b-chat/blob/main/config.json
    dict(
        name="Danube2-1.8b-chat",
        hf_config=dict(org="h2oai", name="h2o-danube2-1.8b-chat"),
        vocab_size=32000,
        n_layer=24,
        n_head=32,
        n_embd=2560,
        block_size=4096,  # should be 8192 but sliding_window mechanism is not implemented
        intermediate_size=6912,
        padding_multiple=64,
        norm_eps=1e-05,
        rope_base=10000,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
    )
]
configs.extend(danube2)


##########################
# Stability AI FreeWilly2
##########################
freewilly_2 = [
    # https://huggingface.co/stabilityai/FreeWilly2/blob/main/config.json
    dict(
        name="FreeWilly2",
        hf_config=dict(org="stabilityai", name="FreeWilly2"),
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
    )
]
configs.extend(freewilly_2)


##################
# Meta Code Llama
##################
code_llama = [
    # https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-Python-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-Instruct-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-Instruct-hf"),
        block_size=2048,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-Instruct-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-Instruct-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
]
configs.extend(code_llama)


########################
# garage-bAInd Platypus
########################
platypus = [
    # https://huggingface.co/garage-bAInd/Platypus-30B/blob/main/config.json
    dict(
        name="Platypus-30B",
        hf_config=dict(org="garage-bAInd", name="Platypus-30B"),
        block_size=2048,
        padded_vocab_size=32000,
        n_layer=60,
        n_head=52,
        n_embd=6656,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-06,
        mlp_class_name="LLaMAMLP",
        intermediate_size=17920,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-7B/blob/main/config.json
    dict(
        name="Platypus2-7B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-7B"),
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-13B/blob/main/config.json
    dict(
        name="Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-13B"),
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-70B/blob/main/config.json
    dict(
        name="Platypus2-70B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-70B"),
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=28672,
    ),
    # https://huggingface.co/garage-bAInd/Camel-Platypus2-13B/blob/main/config.json
    dict(
        name="Camel-Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Camel-Platypus2-13B"),
        padded_vocab_size=32000,
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
    # https://huggingface.co/garage-bAInd/Camel-Platypus2-70B/blob/main/config.json
    dict(
        name="Camel-Platypus2-70B",
        hf_config=dict(org="garage-bAInd", name="Camel-Platypus2-70B"),
        padded_vocab_size=32000,
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
    # https://huggingface.co/garage-bAInd/Stable-Platypus2-13B/blob/main/config.json
    dict(
        name="Stable-Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Stable-Platypus2-13B"),
        padded_vocab_size=32000,
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
    # https://huggingface.co/garage-bAInd/Platypus2-70B-instruct/blob/main/config.json
    dict(
        name="Platypus2-70B-instruct",
        hf_config=dict(org="garage-bAInd", name="Platypus2-70B-instruct"),
        padded_vocab_size=32000,
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
configs.extend(platypus)


##################################
# togethercomputer LLaMA-2-7B-32K
##################################
together_llama2_32k = [
    # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/config.json
    dict(
        name="LLaMA-2-7B-32K",
        hf_config=dict(org="togethercomputer", name="LLaMA-2-7B-32K"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=8,
    )
]
configs.extend(together_llama2_32k)


################
# Microsoft Phi
################
phi = [
    # https://huggingface.co/microsoft/phi-1_5/blob/main/config.json
    dict(
        name="phi-1_5",
        hf_config=dict(org="microsoft", name="phi-1_5"),
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2048,
        n_layer=24,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
    # https://huggingface.co/microsoft/phi-2/blob/main/config.json
    dict(
        name="phi-2",
        hf_config=dict(org="microsoft", name="phi-2"),
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2560,
        n_layer=32,
        rotary_percentage=0.4,  # 32 / (n_embd / n_head) = 32 / 80
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
]
configs.extend(phi)


#############
# Mistral AI
#############
mistral = [
    # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
    dict(
        name="Mistral-7B-{}v0.1",
        hf_config=dict(org="mistralai", name="Mistral-7B-{}v0.1"),
        padded_vocab_size=32000,
        block_size=4096,  # should be 32768 but sliding window attention is not implemented
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    ),
    # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
    dict(
        name="Mixtral-8x7B-{}v0.1",
        hf_config=dict(org="mistralai", name="Mixtral-8x7B-{}v0.1"),
        padded_vocab_size=32000,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMoE",
        intermediate_size=14336,
        rope_base=1000000,
        n_expert=8,
        n_expert_per_token=2,
    ),
]
for c in mistral:
    for kind in ("", "Instruct-"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

configs.append(  # v0.2 foundational not released by Mistral on HF...
    # https://huggingface.co/unsloth/mistral-7b-v0.2/blob/main/config.json
    dict(
        name="Mistral-7B-v0.2",
        hf_config=dict(org="unsloth", name="Mistral-7B-v0.2"),
        padded_vocab_size=32000,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    )
)
configs.append(
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json
    dict(
        name="Mistral-7B-Instruct-v0.2",
        hf_config=dict(org="mistralai", name="Mistral-7B-Instruct-v0.2"),
        padded_vocab_size=32000,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    )
)
configs.append(
    # https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/main/config.json
    dict(
        name="Mistral-7B-v0.3",
        hf_config=dict(org="mistralai", name="Mistral-7B-v0.3"),
        padded_vocab_size=32768,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    )
)
configs.append(
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json
    dict(
        name="Mistral-7B-Instruct-v0.3",
        hf_config=dict(org="mistralai", name="Mistral-7B-Instruct-v0.3"),
        padded_vocab_size=32768,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=14336,
    )
)



############
# TinyLlama
############
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


##########################
# Trelis Function Calling
##########################
llama_2_function_calling = [
    # https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2/blob/main/config.json
    dict(
        name="Llama-2-7b-chat-hf-function-calling-v2",
        hf_config=dict(org="Trelis", name="Llama-2-7b-chat-hf-function-calling-v2"),
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        norm_eps=1e-6,
        block_size=4096,
        vocab_size=32000,
        n_head=32,
        n_embd=4096,
        rope_base=10000,
    )
]

configs.extend(llama_2_function_calling)

name_to_config = {config["name"]: config for config in configs}
