#!/usr/bin/env python3

from contextlib import nullcontext

import torch

"""
Configuration file - GPT

### Parameters description:
    BLOCK_SIZE: context length (max distance between 2 tokens over which 
        attention is evaluated)
    BATCH_SIZE: batch size
    N_EMBD: dimension of the embedding (token representation) - [$d_{model}$]
    N_HEADS: number of attention heads - [$n_{heads}$]
        "Head size" [$d_{head}$] = N_EMBD / N_HEADS: size of Q, K, V
    N_LAYER: number of layers in the transformer (n.times block is repeated) - 
        [$n_{layers}$]
    DROPOUT: fraction of dropped connections
    N_ITER_TRAIN: number of training iterations (epochs)
    LEARNING_RATE: [...]
    EVAL_INTERVAL: epochs between each loss evaluation
    EVAL_ITERS: number of iterations over which loss is averaged
    DEVICE: PyTorch device used for training ('cuda'/'mps'/'cpu')
"""

# ---- Model configuration -----------------------
BLOCK_SIZE = 128  # (context length in chars) - affects VRAM
BATCH_SIZE = 24  # affects VRAM (if gr. acc. st > 1, it's the micro-batch size)
N_EMBD = 384  # Number of token embeddings processed at each time instant
N_HEADS = 6  # Number of attention heads (NOTE: head size = 384 / 6 = 64)
N_LAYER = 9  # Number of transformer blocks (TRAINING ONLY)
DROPOUT = 0.2  # Dropout probability
BIAS = True  # do we use bias inside LayerNorm and Linear layers?

GRADIENT_ACCUMULATION_STEPS = 5 * 8  # used to simulate larger batch sizes

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# DEVICE = "cpu"  # On macOS there are some issues with MPS

# ---- Training configuration --------------------
INIT_FROM = "resume"  # "scratch" or "resume" ("gpt2" not implemented)

MAX_ITERS = N_ITER_TRAIN = 600000  # total number of training iterations
# Loss evaluation
CKPT_INTERVAL = 2000
EVAL_ITERS = 200
LOG_INTERVAL = 10
EVAL_ONLY = False  # if True, script exits right after the first eval
ALWAYS_SAVE_CHECKPOINT = True  # T: always save a checkpoint after each eval

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
TOP_K = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
TEMPERATURE = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions

# ---- MDI settings ------------------------------
# Adaptive layer number - first key is the number of nodes, second is the node
# type
N_LAYERS_NODES = {
    2: {
        5: {"N_LAYERS_START": 2, "N_LAYERS_FINISH": 3},
        7: {"N_LAYERS_START": 3, "N_LAYERS_FINISH": 4},
        9: {"N_LAYERS_START": 4, "N_LAYERS_FINISH": 5},
        12: {"N_LAYERS_START": 5, "N_LAYERS_FINISH": 7},
    },
    3: {
        5: {"N_LAYERS_START": 1, "N_LAYERS_INTERM": 2, "N_LAYERS_FINISH": 2},
        7: {"N_LAYERS_START": 1, "N_LAYERS_INTERM": 3, "N_LAYERS_FINISH": 3},
        9: {"N_LAYERS_START": 1, "N_LAYERS_INTERM": 4, "N_LAYERS_FINISH": 4},
        12: {"N_LAYERS_START": 2, "N_LAYERS_INTERM": 5, "N_LAYERS_FINISH": 5},
    },
}

# CTX = (
#     nullcontext()
#     if DEVICE in {"cpu", "mps"}
#     else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# )
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
VERB = True
DEBUG = True
PLOTS = True
