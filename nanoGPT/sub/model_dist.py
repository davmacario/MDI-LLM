#!/usr/bin/env python3

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Union

import cherrypy as cp
import requests
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
original model.

Rationale:
    - Three block types:
        a. Starter: embedding (tok + pos), dropout,
        [b. Intermediate: 2 transformer layers]
        c. Finisher: 2 transformer layers, layer norm
    - The "Starter" is the main node - it will initiate inference and receive 
    the final logits (outputs of the model) from the "Finisher"
    - The Starter is also the only one to possess all the model parameters
    
    - Each node opens 2 ports: one for communication (roles setup and weight
    exchange), the other one for the actual inference (i.e., exchange of
    activations)

    - All configuration information is stored in a .json file

Functioning:
    - All nodes run the webserver - used to exchange control info & parameters
    only
    - Main node instantiates GPTDistributed object, which in turn instantiates
    the StarterNode on the same device
    - Main node (through GPTDistributed) sends HTTP messages about their role to 
    each other node in the network (role + weights + next_node [& maybe
    predecessor])
    - The other nodes instantiate the corresponding object (Intermediate/
    Finisher) and open sockets to/from the corresponding nodes, then start 
    waiting for the information to arrive
"""

N_LAYERS_INTERM = 2  # Number of transformer layers in each intermediate node
N_LAYERS_FINISH = 2


class StateDict(Mapping[str, Any]):
    def __init__(self):
        super().__init__()


class StarterNode(nn.Module):
    """Starter node"""

    params_init = False

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None

        self.config = config

        self.starter_model = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(config.vocab_size, config.n_embd),
                position_embedding=nn.Embedding(
                    config.block_size, config.n_embd
                ),
                drop=nn.Dropout(config.dropout),
            )
        )
        pass

    def load_weights(self, params: StateDict) -> int:
        """Load weights"""
        self.load_state_dict(params)
        self.params_init = True
        return 1

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass - starter"""
        if not self.params_init:
            raise ValueError("The model parameters have not been initialized!")

        device = idx.device

        _, t = idx.shape  # Batch x (Time dimension)
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, as block size (context length) is {self.config.block_size}"
            )

        # The logits returned are the ones in row idx of the table
        # This is arranged in a tensor of size Batch x Time x Channel(=N_EMBED)
        tok_emb = self.starter_model.token_embedding(idx)

        # Obtain positional embeddings by encoding values (0, ..., t)
        pos_emb = self.starter_model.position_embedding(
            torch.arange(t, device=device)
        )

        x = self.starter_model.drop(tok_emb + pos_emb)  # (B, T, C)

        return x


class IntermediateNode(nn.Module):
    """Intermediate node"""

    params_init = False

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None

        self.config = config

        # Follow naming convention
        self.intermediate_model = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [Block(config) for _ in range(N_LAYERS_INTERM)]
                )
            )
        )

    def load_weights(self, params: StateDict) -> int:
        """Load weights"""
        self.load_state_dict(params)
        self.params_init = True
        return 1

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass - intermediate node"""
        if not self.params_init:
            raise ValueError("The model parameters have not been initialized!")
        x = idx
        for block in self.intermediate_model.layers:
            x = block(x)
        return x


class FinisherNode(nn.Module):
    """Finisher node"""

    params_init = False

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None

        self.config = config

        self.finisher_model = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [Block(config) for _ in range(N_LAYERS_FINISH)]
                ),
                ln_f=LayerNorm(config.n_head, bias=config.bias),
                lm_head=nn.Linear(config.n_embd, config.vocab_size),
            )
        )

    def load_weights(self, params: StateDict) -> int:
        """Load weights"""
        self.load_state_dict(params)
        self.params_init = True
        return 1

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass - finisher node"""
        if not self.params_init:
            raise ValueError("The model parameters have not been initialized!")
        x = idx
        for block in self.finisher_model.layers:
            x = block(x)
        x = self.finisher_model.ln_f(x)
        logits = self.finisher_model.lm_head(x)

        return logits


class CommunicationServer:
    """
    Communication server - Cherrypy-based webserver used for exchanging
    (receiving) setup and control information
    """

    exposed = True

    def __init__(self, node_type: str, node_config: Dict):
        self.own_addr = node_config["addr"]
        self.own_comm_port = node_config["communication"]["port"]

        # FIXME: maybe the following configuration should be called outside
        cp.tree.mount(self, "/")
        cp.config.update(
            {
                "server.socket_host": self.own_addr,
                "server.socket_port": self.own_comm_port,
            }
        )
        if node_type.lower() == "starter":
            # Configuration of starter node
            pass
        else:
            # Configuration of "secondary" (intermediate or finisher) node
            self.starter_addr = node_config["communication"]["starter_addr"]
        pass

    def GET(self, *path, **params):
        pass

    def POST(self, *path, **params):
        pass

    def PUT(self, *path, **params):
        pass


def split_parameters(
    model_params: StateDict, n_nodes: int
) -> Dict[str, Mapping[str, Any]]:
    """
    Split the model parameters (contained in a state dict) among the different
    available nodes.

    The number of nodes should be at least 2 (starter and finisher).

    The parameters are divided as such:
        - Starter: token embedding, positional embedding
        - Intermediate: 2xTransformer Layer
        - Finisher: 2xTransformer Layer, LayerNorm
    """
    assert n_nodes >= 2

    out_chunks = {}  # TODO: add same keys as config["nodes"]

    # Do stuff

    return out_chunks


class GPTDistributed:
    def __init__(
        self,
        config: GPTConfig,
        ckpt_path: str,
        nodes_info_path: Union[str, None] = None,
    ):
        """
        Instantiate a GPTDistributed object to perform model-distributed
        inference.

        Args:
            config: GPTConfig object with the relevant model parameters
            ckpt_path: path of the full model (pretrained)
            nodes_info_path: path of the configuration JSON - if not provided,
                a default one is used
        """
        if nodes_info_path is not None:
            settings_path = nodes_info_path
        else:
            settings_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "settings_distr",
                "configuration.json",
            )

        with open(settings_path, "r") as f:
            self.nodes_info = json.load(f)
            f.close()

        # Store important parameters:
        self.own_config = self.nodes_info["nodes"]["starter"]
        self.own_addr = self.own_config["addr"]
        self.own_comm_port = self.own_config["communication"]["port"]
        self.own_inference_port = self.own_config["inference"]["port"]

        # Get the model parameters and split them based on n. of nodes
        self.n_total_nodes = 2 + len(self.nodes_info["nodes"]["intermediate"])
        self.model_ckpt = torch.load(ckpt_path)
        self.complete_model = self.model_ckpt["model"]  # State dict
        self.model_chunks = split_parameters(
            model_params=self.complete_model, n_nodes=self.n_total_nodes
        )

        # Create webserver
        self.webserv = CommunicationServer(
            node_type="starter", node_config=self.own_config
        )

        # Create 'StarterNode' object
        self.config = config
        self.starter_node = StarterNode(config)
        self.starter_node.load_weights(self.model_chunks["starter"])

    def configure_nodes(self):
        """
        Send POST requests to the other nodes to inform them of their role and
        including their chunk of model.
        """
        # Intermediate config
        for int_node in self.nodes_info:
            pass
        # Finisher config
        pass
