# Copyright (c) 2024 Davide Macario
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from sub.model import Block, Config, build_mask_cache, build_rope_cache
from .utils import init_from_state_dict

"""
This file contains the definitions of the different nodes in the MDI architecture.
"""


class NodePrototype(nn.Module):
    """
    This class contains the common methods for all nodes.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.transformer = nn.ModuleDict()
        self.mask_cache: Optional[torch.Tensor] = None
        self.verb = True if "verb" in kwargs and kwargs["verb"] else False

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}"
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot
        # update it here because we don't know if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize KV cache to allow for faster inference.
        """
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def init_rope_mask(
        self,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


# -------------------------------------------------------------------------------------


class StarterNode(NodePrototype):
    """Starter node"""

    params_init = False

    def __init__(
        self,
        config: Config,
        n_transf_layers: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert config.padded_vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Initial layers
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(n_transf_layers)]),
                # Final layers:
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.max_seq_length = self.config.block_size

    def load_weights(self, params: Dict[str, Any], **kwargs) -> int:
        """Load sub-model weights"""
        # self.load_state_dict(params, **kwargs)
        init_from_state_dict(self, params)
        self.params_init = True
        if self.verb:
            print(f"Weights loaded!")
        return 1

    def forward(
        self,
        idx: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *,
        first_pass: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Forward pass - starter

        Args:
            idx: input tensor
            input_pos:
            first_pass: boolean value - if True (default) it indicates the first forward
                pass (embedding and transformer blocks), else it will pass the input
                through the output layers of the LLM (`ln_f`, `lm_head`)
        """
        if first_pass:
            B, T = idx.shape  # Batch x (Time dimension)
            if T > self.config.block_size:
                raise ValueError(
                    f"Cannot forward sequence of length {T}, as block size (context length) is {self.config.block_size}"
                )

            if B > 1:
                raise NotImplementedError("Currently supporting batch sizes = 1 only")

            if input_pos is not None:  # use the kv cache
                cos = self.cos.index_select(0, input_pos)
                sin = self.sin.index_select(0, input_pos)
                if self.mask_cache is None:
                    raise TypeError("You need to call `gpt.set_kv_cache()`")
                mask = self.mask_cache.index_select(2, input_pos)
            else:
                cos = self.cos[:T]
                sin = self.sin[:T]
                mask = None

            # The logits returned are the ones in row idx of the table
            # This is arranged in a tensor of size Batch x Time x Channel(=N_EMBED)
            x = self.transformer.wte(idx)
            if self.config.scale_embeddings:
                x = x * (self.config.n_embd**0.5)

            for block in self.transformer.h:
                x = block(x, cos, sin, mask, input_pos)

            return x
        else:
            idx = self.transformer.ln_f(idx)
            return self.lm_head(idx)


class SecondaryNode(NodePrototype):
    """Secondary worker node"""

    params_init = False

    def __init__(
        self,
        config: Config,
        n_transf_layers: int,
        **kwargs,
    ):
        """
        Args:
            config: Config object with the model setup parameters
            n_transf_layers: number of local transformer layers
            [**kwargs]
        """
        super().__init__(**kwargs)
        assert config.vocab_size is not None
        self.config = config

        # Follow naming convention
        self.transformer = nn.ModuleDict(
            dict(h=nn.ModuleList([Block(config) for _ in range(n_transf_layers)]))
        )
        self.max_seq_length = self.config.block_size

    def load_weights(self, params: Dict[str, Any], **kwargs) -> int:
        """Load weights"""
        # self.load_state_dict(params, **kwargs)
        init_from_state_dict(self, params)
        self.params_init = True
        if self.verb:
            print(f"Weights loaded!")
        return 1

    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass - secondary node"""
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        for block in self.transformer.h:
            idx = block(idx, cos, sin, mask, input_pos)
        return idx
