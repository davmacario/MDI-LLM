#!/usr/bin/env python3

import inspect
import math
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import (BIAS, BLOCK_SIZE, DEVICE, DROPOUT, N_EMBD, N_HEADS,
                     N_LAYER, PLOTS, VERB)

ACT2FN = {"GELU": nn.GELU(), "ReLU": nn.ReLU()}


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply
    bias=False (actually, in newer versions, it does)
    """

    def __init__(self, ndim: int, bias: bool = True):
        """
        Args:
            ndim: dimension of the input (n. weights)
            bias: if true, use bias, else don't
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class GPTConfig:
    """Wrapper for GPT configuration parameters"""

    block_size: int = BLOCK_SIZE  # Context length
    vocab_size: Union[int, None] = 50304  # from GPT-2: 50257 (round to multiple of 64)
    n_layer: int = N_LAYER  # Number of transformer blocks
    n_head: int = N_HEADS
    n_embd: int = N_EMBD
    dropout: float = DROPOUT
    bias: bool = BIAS  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    activation_function: str = "GELU"  # Or ReLU

    def asdict(self):
        return {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
        }


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Attention module: use multiple attention heads in parallel.

    NOTE: causal attention only (only eval. attention scores with previous tokens)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert (
            config.n_embd % config.n_head == 0
        ), f"Invalid combination of embedding length ({config.n_embd}) and number of attention heads per layer ({config.n_head})"

        # K, Q, V projections - all heads in batch (no need to compute separate Q, K, V)
        # TODO: switch from Linear to Conv1D, as in Transformers (edit from_pretrained)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # NOTE: flash attention unavailable on Jetson TX2 - only if Torch >= 2.0 (I'm on 1.12)
        self.flash = False
        if not self.flash:
            # if VERB:
            #     print("Using slow attention - flash attention not available")
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones((config.block_size, config.block_size), dtype=torch.bool)
                ).view(1, 1, config.block_size, config.block_size),
                # persistent=False,  # Prevent from saving the buffer to .pt (not in state_dict)
            )
        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
        #         1, 1, max_positions, max_positions
        #     ),
        #     persistent=False,
        # )

    def forward(self, x):
        """
        Forward pass, multi-head attention.

        First, evaluate the scaled self-attention matrix multiplying queries and keys.
        Then, mask the matrix using the triangular matrix - this implies that attention
        scores are only evaluated between a token and its predecessors (no relationship
        is accounted for with the following ones).

        Lastly, the masked matrix is used to multiply the values, returning the output -
        the weighted sum of all values by the attention score.

        Notice that the size of the time dimension (T) changes based on the context
        length.
        In particular, at the beginning of generation, when the number of tokens is
        lower than the contex length, T will be lower than the context length, and the
        triangular mask will be truncated.
        """
        B, T, C = x.size()  # Batch sz., Seq. len, Embedding dim.
        assert C == self.n_embd

        # Q, K, V (altogether in batch), then change shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape to: (B, n_head, T, head_size) [head_size = n_embd//n_head]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention - output size: (B, nh, T, T)
        if self.flash:
            raise ValueError()
        else:
            hs = q.shape[-1]  # Normalization factor (head size - len of Q, K, V)
            # (B, nh, T, hs) @ (B, nh, hs, T):
            wei = (q @ k.transpose(-2, -1)) * (hs**-0.5)
            wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
            wei = self.attn_dropout(wei)
            y = wei @ v

        # Concatenate head outputs together (and make it the same size as input)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection
        out = self.resid_dropout(self.c_proj(y))
        return out


class MLP(nn.Module):
    """Simple multilayer perceptron, used after MHA"""

    def __init__(self, config: GPTConfig):
        """
        Create feed-forward layer used after MHA.

        Args:
            n_embd: number of input and output embeddings (i.e., neurons)
        """
        super().__init__()
        # The 4* comes from the Transformer paper + the final linear layer is
        # necessary for residual connections & it brings the dimension down
        # to the value we want (n_embd)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = ACT2FN[config.activation_function]
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block - communication + computation

    Note: decoder only
    """

    def __init__(self, config):
        """
        Instantiate Transformer block

        Args:
            n_embd: number of token embeddings per time batch
            n_head: number of attention heads (must be a divisor of n_embd, as
                each head will work with dim. n_embd // n_head)
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        # NOTE: using residual connections (sum the inputs to the outputs)
        # LayerNorm applied before each layer
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT2 implementation"""

    def __init__(self, config: GPTConfig, **setup):
        """
        Create GPT object.

        Args:
            config: GPTConfig object with all the configuration parameters
            setup: dict containing overrides for global constants
        """
        assert config.vocab_size is not None

        if "plots" in setup:
            global PLOTS
            PLOTS = setup["plots"]
        if "verb" in setup:
            global VERB
            VERB = setup["verb"]

        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config: GPTConfig = config

        self.transformer = nn.ModuleDict(
            dict(
                # Each token will read the logits for the next token from a lookup table
                # inputs dim: vocab_size (tokens)
                # outputs dim: n_embd
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Positional embedding
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # Dropout layer before transformer layers
                drop=nn.Dropout(config.dropout),
                # Multi-Head Attention + FC layers
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        # Output linear layer, producing logits before softmax (only important at training)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight-Tying: share weights of embedding and output layers
        self.transformer.wte.weight = self.lm_head.weight

        # Initialization
        self.apply(self._init_weights)

        # NOTE: from original implementation:
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        if VERB:
            print(f"Number of parameters: {self.get_num_params()}")

    def _init_weights(self, module):
        """
        Initialize the model parameters.

        The function is applied to each module defined at instantiation.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get
        subtracted.
        The token embeddings would too, except due to the parameter sharing
        these params are actually used as weights in the final layer, so we
        include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor, targets: Union[torch.Tensor, None] = None):
        """
        Forward pass in GPT.

        If a target is provided (at training), this method also evaluates the
        loss.

        Args:
            idx: (Batch size) x (Time) tensor of integers
            targets: target embedding, same size as idx; default: None

        Returns:
            logits - row 'idx' of the token embedding table; size is
                B x T x vocab_size
        """
        device = idx.device

        # Need to ensure the sequence of embeddings <= context length
        _, t = idx.shape  # Batch x Time x 1 (1 token/time position)
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, as max. block size (context length) is {self.config.block_size}"
            )

        # The logits returned are the ones in row idx of the table
        # This is arranged in a tensor of size Batch x Time x Channel(=N_EMBED)
        tok_emb = self.transformer.wte(idx)
        # Obtain positional embeddings by encoding values (0, ..., t)
        pos_emb = self.transformer.wpe(torch.arange(t, device=device))
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (B, T, C)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # Conform to PyTorch's specs - fix dimensions
            # In this case, flatten the first and second dimensions of 'logits'
            # (B, T, C) --> (B * T, C)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Only forward the lm_head on the last position - slight optimization
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """
        Perform model surgery to decrease the block size if necessary;
        E.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        but want to use a smaller block size for some smaller, simpler model

        Args:
            block_size: new (smaller) block size
        """
        assert (
            block_size <= self.config.block_size
        ), f"Need to provide a smaller block size than {self.config.block_size}"

        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type: str, override_args=None):
        """
        Load weights from external pretrained models.

        Args:
            model_type: string indicating the model type, must be one of:
                "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
            override_args

        Returns:
            model using the loaded parameters
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [
            k for k in sd.keys() if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
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
        # Openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Set up optimizers used for training the model.

        Args:
            weight_decay
            learning_rate
            betas: tuple containing the hyperparameters of Adam optimizer.
            device_type: type of the available device for training
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak
        FLOPS
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd // cfg.n_head,
            cfg.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Union[int, None] = None,
    ) -> Tuple[torch.Tensor, List]:
        """
        Generate new tokens using GPT, provided the input sequence of integers
        "idx".

        If generating text (inference), make sure to use the model in "eval" mode
        (`model.eval()`).

        Args:
            idx: input sequence of encoded chars/words (B x T) - most likely
                [[0]]
            max_new_tokens: maximum number of tokens to be generated
            temperature: scaling of logits before softmax - hyperparam.,
            default: 1 (NOTE: if high, more "pointy" distribution, hence more "sure")

        Returns:
            Updated version of idx, containing the generated elements
            Total generation time in seconds (more accurate than timing this function)
        """
        from .utils import loading_bar

        t_start = time.time()
        tok_time = []

        for i in range(max_new_tokens):
            tok_time.append((i, time.time() - t_start))
            print(
                f"Generating {loading_bar(i, max_new_tokens, 30)} {i}/{max_new_tokens}",
                end="\r",
            )
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
            # Optionally crop the logits to only the top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Apply softmax to get probabilities of tokens in the last time step
            probs = F.softmax(logits, dim=1)
            # Sample p.m.f, get output token representation (need to decode)
            idx_next = torch.multinomial(probs, num_samples=1)  # B x 1
            # Append sampled index to idx to generate next sample
            idx = torch.cat((idx, idx_next), dim=1)  # B x (T+1)

        tot_gen_time = time.time() - t_start
        tok_time.append((max_new_tokens, tot_gen_time))
        print("\nGeneration completed!")
        return (idx, tok_time)
