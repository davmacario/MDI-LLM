#!/usr/bin/env python3

import inspect
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Union

import cherrypy as cp
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.config import (BATCH_SIZE, BIAS, BLOCK_SIZE, CKPT_INTERVAL, DEVICE,
                        DROPOUT, EVAL_ITERS, LEARNING_RATE, N_EMBD, N_HEADS,
                        N_ITER_TRAIN, N_LAYER, VERB)
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


def remove_prefix(text: str, prefix: str) -> str:
    """
    Remove the specified prefix from the given string.
    NOTE: starting Python 3.9, use text.removeprefix(prefix)

    Args:
        text
        prefix

    Returns:
        if the prefix is present, the string without it, else the full given
        string
    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def split_parameters(
    model_params: Mapping[str, Any], n_nodes: int
) -> Dict[str, Any]:
    """
    Split the model parameters (contained in a state dict) among the different
    available nodes.

    The number of nodes should be at least 2 (starter and finisher).

    The parameters are divided as such:
        - Starter: token embedding, positional embedding
        - Intermediate: 2xTransformer Layer
        - Finisher: 2xTransformer Layer, LayerNorm

    Args:
        model_params: complete model parameters (state dict)
        n_nodes: number of nodes among which to divide the parameters; must be
        greater or equal to 2 (at least starter and finisher)

    Returns:
        dict containing the following k-v pairs:
            "starter": dict with the starter state dict
            "intermediate": list containing the intermediate state dicts
            "finisher": dict with the finisher state dict
    """
    # TODO: make more efficient (too many nested loops) - maybe
    # Set up some parameters - they are used to gather the relevant keys
    base_name_transformer = "transformer"
    tok_emb = "token_embedding"
    pos_emb = "position_embedding"
    layer_name = "layers"
    transformer_last = f"{base_name_transformer}.ln_f"
    output_layer = "lm_head"

    assert n_nodes >= 2

    out_chunks = {}  # TODO: add same keys as config["nodes"]

    # 1. Select params for Starter
    out_chunks["starter"] = {}
    out_chunks["starter"][f"starter_model.{tok_emb}.weight"] = model_params[
        f"{base_name_transformer}.{tok_emb}.weight"
    ]
    out_chunks["starter"][f"starter_model.{pos_emb}.weight"] = model_params[
        f"{base_name_transformer}.{pos_emb}.weight"
    ]
    try:
        out_chunks["starter"][f"starter_model.{tok_emb}.bias"] = model_params[
            f"{base_name_transformer}.{tok_emb}.bias"
        ]
    except:
        # Here if no bias - no problem
        pass

    try:
        out_chunks["starter"][f"starter_model.{pos_emb}.bias"] = model_params[
            f"{base_name_transformer}.{pos_emb}.bias"
        ]
    except:
        # Here if no bias - no problem
        pass

    # 2. Select params for every Intermediate
    out_chunks["intermediate"] = []
    for i in range(1, n_nodes - 1):
        curr_params = {}

        # Complicated pythonic list call to select the correct keys to be
        # transferred to the intermediate node
        # As reference, the keys for the layers all start with:
        #       transformer.layer.<layer_ind>.[...]
        # so we need to select the correct layer indices
        valid_layer_ind = list(
            range((i - 1) * N_LAYERS_INTERM, i * N_LAYERS_INTERM)
        )
        relevant_keys = [
            k
            for k in list(model_params.keys())
            if (
                k.startswith(f"{base_name_transformer}.{layer_name}")
                and int(k.split(".")[2]) in valid_layer_ind
            )
        ]

        # Iterate over old keys, select correct ones, create new keys, transfer values
        local_layer_ind = 0
        for ind in valid_layer_ind:
            prefix = f"{base_name_transformer}.{layer_name}.{ind}"
            for k in relevant_keys:
                if k.startswith(prefix):
                    new_k = f"intermediate_model.layers.{local_layer_ind}.{remove_prefix(k, prefix)}"
                    curr_params[new_k] = model_params[k]
            local_layer_ind += 1

        out_chunks["intermediate"].append(curr_params)

    # 3. Select params for Finisher
    out_chunks["finisher"] = {}

    # Layers:
    valid_layer_ind = list(
        range((n_nodes - 1) * N_LAYERS_FINISH, n_nodes * N_LAYERS_FINISH)
    )
    relevant_keys = [
        k
        for k in list(model_params.keys())
        if (
            k.startswith(f"{base_name_transformer}.{layer_name}")
            and int(k.split(".")[2]) in valid_layer_ind
        )
    ]
    local_layer_ind = 0
    for ind in valid_layer_ind:
        prefix = f"{base_name_transformer}.{layer_name}.{ind}"
        for k in relevant_keys:
            if k.startswith(prefix):
                new_k = f"finisher_model.layers.{local_layer_ind}.{remove_prefix(k, prefix)}"
                out_chunks["finisher"][new_k] = model_params[k]
        local_layer_ind += 1

    out_chunks["finisher"][f"finisher_model.ln_f.weight"] = model_params[
        f"{transformer_last}.weight"
    ]
    try:
        out_chunks["finisher"][f"finisher_model.ln_f.bias"] = model_params[
            f"{transformer_last}.bias"
        ]
    except:
        pass

    out_chunks["finisher"][f"finisher_model.lm_head.weight"] = model_params[
        f"{output_layer}.weight"
    ]
    try:
        out_chunks["finisher"][f"finisher_model.lm_head.weight"] = model_params[
            f"{output_layer}.weight"
        ]
    except:
        pass

    return out_chunks


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

    def load_weights(self, params: Mapping[str, Any]) -> int:
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

    def load_weights(self, params: Mapping[str, Any]) -> int:
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

    def load_weights(self, params: Mapping[str, Any]) -> int:
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


# -----------------------------------------------------------------------------


class GPTServer:
    """
    Communication server - Cherrypy-based webserver used for exchanging
    (receiving) setup and control information
    """

    # TODO: add model initialization for the different node types

    exposed = True
    model: Union[None, StarterNode, IntermediateNode, FinisherNode] = None
    next_node: Union[None, Dict] = None
    prev_node: Union[None, Dict] = None
    model_params: Union[None, Dict] = None
    model_config: Union[None, GPTConfig] = None

    def __init__(
        self,
        node_type: str,
        node_config: Dict,
        **kwargs,
    ):
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

        self.node_type = node_type
        self.node_config = node_config
        if node_type.lower() == "starter":
            # Configuration of starter node
            assert len(kwargs) >= 3  # There should be params, next & prev

            if "next_node" in kwargs:  # FIXME: check this works
                self.next_node = dict(kwargs["next_node"])
            else:
                raise ValueError("Missig 'next_node' information")

            if "prev_node" in kwargs:  # FIXME: check this works
                self.prev_node = dict(kwargs["prev_node"])
            else:
                raise ValueError("Missig 'prev_node' information")

            if "params" in kwargs:  # FIXME: check this works
                self.model_params = dict(kwargs["params"])
            else:
                raise ValueError("Missig 'prev_node' information")

            if "model_config" in kwargs:
                # Create object as done in train/sample scripts
                self.model_config = GPTConfig(**kwargs["model_config"])
            else:
                raise ValueError("Missing model configuration")

            self.init_model()

        else:
            # Configuration of "secondary" (intermediate or finisher) node
            self.starter_addr = node_config["communication"]["starter_addr"]
        pass

    # ----- PRIVATE -----------------------------------------------------------

    def init_model(self):
        """
        Initialize the node's model chunk, passing the parameters, the next and
        previous nodes in the network
        """
        # NOTE: use self.
        assert self.model_params is not None, "No model parameters were found!"
        assert self.model is None, "The model was already initialized!"
        assert (
            self.model_config is not None
        ), "No model configuration was found!"

        if self.node_type == "starter":
            self.model = StarterNode(self.model_config)

    # ----- PUBLIC ------------------------------------------------------------

    def GET(self, *path, **params):
        """
        Functions
            Return node information (port numbers, [capabilities]?)
            Used for pinging "neighbor" nodes
        """
        if len(path) == 0:
            return json.dumps(self.node_config)

    def POST(self, *path, **params):
        """
        Functions:
        - Non-starters:
            Receive configuration info from the starter node and start
            connection, with previous and next, then wait for incoming
            transmission (from prev), process values (forward), and send them
            over to the next one
        """
        pass

    def PUT(self, *path, **params):
        """
        Functions:
            ?
        """
        pass


class GPTDistributed:
    """
    Distributed implementation of a minimal GPT2 instance
    """

    # Syntax of the message used to initialize other nodes
    init_msg = {
        "role": "",
        "params": {},
        "model_config": {}
        "prev_node": {"addr": "", "port": 8088},  # NOTE: INFERENCE port
        "next_node": {"addr": "", "port": 8088},
    }

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
        self.n_intermediate = len(self.nodes_info["nodes"]["intermediate"])
        self.n_total_nodes = 2 + self.n_intermediate
        try:
            self.model_ckpt = torch.load(ckpt_path, map_location=DEVICE)
        except:
            # It may be that the model does not fit all in the device
            self.model_ckpt = torch.load(ckpt_path, map_location="cpu")
        self.complete_model = self.model_ckpt["model"]  # State dict
        self.model_chunks = split_parameters(
            model_params=self.complete_model, n_nodes=self.n_total_nodes
        )

        self.config = config

        # Create webserver
        starter_config = self.init_msg.copy()
        del starter_config["role"]  # FIXME: necessary?
        starter_config["params"] = self.model_chunks["starter"]
        starter_config["model_config"] = self.config.asdict()
        starter_config["prev_node"] = {
            "addr": self.nodes_info["finisher"]["addr"],
            "port": self.nodes_info["finisher"]["inference"]["port"],
        }
        if self.n_intermediate:
            starter_config["next_node"] = {
                "addr": self.nodes_info["intermediate"][0]["addr"],
                "port": self.nodes_info["intermediate"][0]["inference"]["port"],
            }
        else:
            starter_config["next_node"] = starter_config["prev_node"]

        self.webserv = GPTServer(
            node_type="starter",
            node_config=self.own_config,
            **starter_config,
        )

    def configure_nodes(self):
        """
        Send POST requests to the other nodes to inform them of their role and
        including their chunk of model.
        """
        # TODO: add also communication information to prev/next node information
        # This way it is possible to ping the nodes (check alive)

        # Store the prev and next in a smart way
        prev = self.nodes_info["starter"]
        n_interm = len(self.nodes_info["intermediate"])
        if n_interm == 0:
            next = prev
        elif n_interm == 1:
            next = self.nodes_info["finisher"]
        else:
            next = self.nodes_info["intermediate"][1]

        # Intermediate config
        for i, int_node in enumerate(self.nodes_info["intermediate"]):
            curr_msg = self.init_msg.copy()
            curr_msg["role"] = "intermediate"
            curr_msg["model_config"] = self.config.asdict()
            curr_msg["params"] = self.model_chunks["intermediate"][i]

            curr_msg["prev_node"]["addr"] = prev["addr"]
            curr_msg["prev_node"]["port"] = prev["inference"]["port"]
            curr_msg["next_node"]["addr"] = next["addr"]
            curr_msg["next_node"]["port"] = next["inference"]["port"]

            # Update next and prev for next iteration
            prev = int_node
            if i == n_interm - 1:
                next = self.nodes_info["starter"]
            elif i == n_interm - 2:
                next = self.nodes_info["finisher"]
            else:
                next = self.nodes_info["intermediate"][i + 2]

            # Send POST request
            target_addr = int_node["addr"]
            target_port = int_node["communication"]["port"]

            addr = f"http://{target_addr}:{target_port}/init"
            ret = requests.post(addr, json=curr_msg)
            while not ret.status_code == 200:
                if VERB:
                    print("Unable to reach node - retrying in 3s")
                time.sleep(3)
                ret = requests.post(addr, json=curr_msg)

        # Finisher config - can use next/prev from last loop iteration
        curr_msg = self.init_msg.copy()
        curr_msg["role"] = "finisher"
        curr_msg["model_config"] = self.config.asdict()

        curr_msg["prev_node"]["addr"] = prev["addr"]
        curr_msg["prev_node"]["port"] = prev["inference"]["port"]
        curr_msg["next_node"]["addr"] = next["addr"]
        curr_msg["next_node"]["port"] = next["inference"]["port"]

        target_addr = self.nodes_info["finisher"]["addr"]
        target_port = self.nodes_info["finisher"]["communication"]["port"]
        addr = f"http://{target_addr}:{target_port}/init"
        ret = requests.post(addr, json=curr_msg)
        while not ret.status_code == 200:
            if VERB:
                print("Unable to reach node - retrying in 3s")
            time.sleep(3)
            ret = requests.post(addr, json=curr_msg)

        return 1

    def start(self):
        """
        Start the operation - webserver + model

        Stop when the model finished running, i.e., all tokens have been
        generated
        """
        pass
