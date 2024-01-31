#!/usr/bin/env python3

import inspect
import os
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.config import (BATCH_SIZE, BIAS, BLOCK_SIZE, CKPT_INTERVAL, DEVICE,
                        DROPOUT, EVAL_ITERS, LEARNING_RATE, N_EMBD, N_HEADS,
                        N_ITER_TRAIN, N_LAYER)
from sub.model import (Block, FeedForward, GPTConfig, Head, LayerNorm,
                       MultiHeadAttention)

"""
Distributed implementation of nanoGPT - using the same blocks defined in the 
original model
"""


class GPTDistributed(nn.Module):
    pass
