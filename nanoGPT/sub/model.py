#!/usr/bin/env python3

import inspect
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.config import (BATCH_SIZE, BIAS, BLOCK_SIZE, DEVICE, DROPOUT,
                        EVAL_INTERVAL, EVAL_ITERS, LEARNING_RATE, N_EMBD,
                        N_HEADS, N_ITER_TRAIN, N_LAYER)


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

    # block_size: int = 1024  # Context length
    # vocab_size: int = 50304  # from GPT-2: 50257 (round to multiple of 64)
    # n_layer: int = 12  # Number of transformer blocks
    # n_head: int = 12
    # n_embd: int = 768
    # dropout: float = 0.0
    # bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    batch_size: int = BATCH_SIZE  # FIXME: it wasn't here before, see if needed
    block_size: int = BLOCK_SIZE  # Context length
    vocab_size: int | None = (
        50304  # from GPT-2: 50257 (round to multiple of 64)
    )
    n_layer: int = N_LAYER  # Number of transformer blocks
    n_head: int = N_HEADS
    n_embd: int = N_EMBD
    dropout: float = DROPOUT
    bias: bool = BIAS  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device: str = DEVICE  # FIXME: new - where to put this?


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
        self.config: GPTConfig = config
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
                mha=nn.ModuleList(
                    [
                        Block(config.n_embd, config.n_head)
                        for _ in range(config.n_layer)
                    ]
                ),
                # mha=nn.Sequential(
                #     *[
                #         Block(config.n_embd, config.n_head)
                #         for _ in range(config.n_layer)
                #     ]
                # ),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        # Output linear layer, producing logits before softmax
        self.lm_head = nn.Linear(N_EMBD, config.vocab_size)

        # Weight-Tying: share weights of embedding and output layers
        self.transformer.token_embedding.weight = self.lm_head.weight

        # Initialization
        self.apply(self._init_weights)

        # FIXME: needed here? it depends on whether the device is in the config
        self = self.to(self.config.device)

        # NOTE: from original implementation:
        # # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.position_embedding.weight.numel()
        return n_params

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type
    ):
        """
        TODO
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
        fused_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Forward pass in GPT.

        If a target is provided (at training), this method also evaluates the
        loss.

        Args:
            idx: (Batch size) x (Time) tensor of integers
            targets: target embedding, same size as idx; default: None

        Returns:
            logits - row 'idx' of the token embedding table; size is
                BxTxvocab_size
        """
        device = idx.device

        _, t = idx.shape  # Batch x (Time dimension)
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, as block size (context length) is {self.config.block_size}"
            )

        # The logits returned are the ones in row idx of the table
        # This is arranged in a tensor of size Batch x Time x Channel(=N_EMBED)
        tok_emb = self.transformer.token_embedding(idx)

        # Obtain positional embeddings by encoding values (0, ..., t)
        pos_emb = self.transformer.position_embedding(
            torch.arange(t, device=device)
        )

        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C)
        # x = self.transformer.mha(x)  # (B, T, C)
        # Fix use of MHA - using nn.ModuleList
        for block in self.transformer.mha:
            x = block(x)
        x = self.transformer.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            # Conform to PyTorch's specs - fix dimensions
            # In this case, flatten the first and second dimensions of 'logits'
            # (B, T, C) --> (B * T, C)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        else:
            # NOTE: missing optimization - in the original one if target is None
            # we only return the last logit (new generated token to be
            # softmaxed)
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
        self.transformer.position_embedding.weight = nn.Parameter(
            self.transformer.position_embedding.weight[:block_size]
        )
        # FIXME: does this only work if the bias is used??
        for block in self.transformer.mha:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[
                    :, :, :block_size, :block_size
                ]

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
            "gpt2-medium": dict(
                n_layer=24, n_head=16, n_embd=1024
            ),  # 350M params
            "gpt2-large": dict(
                n_layer=36, n_head=20, n_embd=1280
            ),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args[
            "vocab_size"
        ] = 50257  # always 50257 for GPT model checkpoints
        config_args[
            "block_size"
        ] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
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

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Generate new tokens using GPT, provided the input sequence of integers
        "idx".

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
            # From original: optionally crop the logits to only the top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Apply softmax to get probabilities of tokens in the last time step
            probs = F.softmax(logits, dim=1)
            # Sample p.m.f, get output token representation (need to decode)
            idx_next = torch.multinomial(probs, num_samples=1)  # B x 1
            # Append sampled index to idx to generate next sample
            idx = torch.cat((idx, idx_next), dim=1)  # B x (T+1)

        return idx
