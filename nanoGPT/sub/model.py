#!/usr/bin/env python3

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.config import (BATCH_SIZE, BLOCK_SIZE, DEVICE, DROPOUT, EVAL_INTERVAL,
                        EVAL_ITERS, LEARNING_RATE, N_EMBD, N_HEADS,
                        N_ITER_TRAIN, N_LAYER)


class Head(nn.Module):
    """Single self-attention head"""

    def __init__(self, head_size: int):
        """
        Instantiate single attention head.

        Args:
            head_size: size of the Query, Key and Value vectors [$d_{head}$]
        """
        super().__init__()

        # Linear projection
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # 'tril' is not a parameter - assign it to a buffer
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor):
        """Forward pass, single attention head"""
        _, T, _ = x.shape  # (B, T, C)
        k = self.key(x)  # (B, T, hs) - hs: "head size"
        q = self.query(x)  # (B, T, hs)

        # Compute attention scores
        # Scaled self-attention
        hs = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (hs**-0.5)  # (B, T, hs) @ (B, hs, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Weighted aggregation
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module: use multiple attention heads in parallel.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        n_embd = num_heads * head_size

        # Create a Module List containing all the heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection - linear transform. of the output of attention heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # The output is the concatenation of the outputs from each head
        # Concat. on "last" dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple linear layer, used after MHA"""

    def __init__(self, n_embd):
        """
        Create feed-forward layer.

        Args:
            n_embd: number of input and output embeddings (i.e., neurons)
        """
        super().__init__()
        # The 4* comes from the Transformer paper + the final linear layer is
        # necessary for residual connections & it brings the dimension down
        # to the value we want (n_embd)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Projection layer
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block - communication + computation

    Note: decoder only
    """

    def __init__(self, n_embd, n_head):
        """
        Instantiate Transformer block

        Args:
            n_embd: number of token embeddings per time batch
            n_head: number of attention heads (must be a divisor of n_embd, as
                each head will work with dim. n_embd // n_head)
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # LayerNorm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor):
        # NOTE: using residual connections (sum the inputs to the outputs)
        # LayerNorm applied before each layer
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    """Wrapper for GPT configuration parameters"""

    block_size: int = 1024
    vocab_size: int = 50304  # from GPT-2: 50257 (round to multiple of 64)
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    """GPT implementation"""

    def __init__(self, config: GPTConfig):
        """
        Create GPT object.

        Args:
            config: GPTConfig object with all the configuration parameters
        """
        assert config.vocab_size is not None
        assert config.block_size is not None

        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                # Each token will read the logits for the next token from a lookup table
                token_embedding=nn.Embedding(config.vocab_size, config.n_embd),
                # Positional embedding
                position_embedding=nn.Embedding(
                    config.block_size, config.n_embd
                ),
                # Dropout layer before MHA
                drop=nn.Dropout(config.dropout),
                # Multi-Head Attention
                mha=nn.Sequential(
                    *[
                        Block(config.n_embd, config.n_head)
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        # Output linear layer, producing logits before softmax
        self.lm_head = nn.Linear(N_EMBD, config.vocab_size)

        # Weight-Tying: share weights of embedding and output layers
        self.transformer.token_embedding.weights = self.lm_head.weights

        # Initialization
        self.apply(self._init_weights)

        # NOTE: from original implementation:
        # # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True):
        """Return the number of parameters of the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        else:
            return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Forward pass in bigram embedding.

        Args:
            idx: (Batch size) x (Time) tensor of integers
            targets: target embedding, same size as idx

        Returns:
            logits - row 'idx' of the token embedding table; size is
                BxTxvocab_size
        """
        B, T = idx.shape

        # The logits returned are the ones in row idx of the table
        # This is arranged in a tensor of size Batch x Time x Channel(=N_EMBED)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            # Conform to PyTorch's specs
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ):
        """
        Generate new tokens using the Bigram Language Model, provided the input
        sequence of integers "idx".

        Args:
            idx: input sequence of encoded chars/words (B x T) - most likely
                [[0]]
            max_new_tokens: maximum number of tokens to be generated
            temperature: scaling of logits before softmax - hyperparam.,
                default: 1

        Returns:
            Updated version of idx, containing the generated elements
        ---
        Note: compared to microGPT, here the option to crop the logits is
        missing
        """
        for _ in range(max_new_tokens):
            # NOTE: crop idx to the last 'block_size' tokens if too long
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # Get predictions - forward
            logits, _ = self(idx_cond)
            # Focus only on last time step (dim: (B, C)), scale by temperature
            logits = logits[:, -1, :] / temperature  # B x C
            # Apply softmax to get probabilities of tokens in the last time step
            probs = F.softmax(logits, dim=1)
            # Sample p.m.f, get output token representation (need to decode)
            idx_next = torch.multinomial(probs, num_samples=1)  # B x 1
            # Append sampled index to idx to generate next sample
            idx = torch.cat((idx, idx_next), dim=1)  # B x (T+1)

        return idx
