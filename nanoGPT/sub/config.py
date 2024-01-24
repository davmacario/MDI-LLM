#!/usr/bin/env python3

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

BLOCK_SIZE = 128  # (context length in chars) - affects VRAM
BATCH_SIZE = 64  # affects VRAM
N_EMBD = 384  # Number of token embeddings processed at each time instant
N_HEADS = 6  # Number of attention heads (head size = 384 / 6 = 64)
N_LAYER = 6  # Number of transformer blocks
DROPOUT = 0.2  # Dropout probability
LEARNING_RATE = 3e-4

N_ITER_TRAIN = 10000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
