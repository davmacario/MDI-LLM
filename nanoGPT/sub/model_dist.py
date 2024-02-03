#!/usr/bin/env python3

import inspect
import json
import os
import pickle
import socket
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Union

import cherrypy as cp
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.char_tokenizer import CharacterTokenizer
from sub.config import (BATCH_SIZE, BIAS, BLOCK_SIZE, CKPT_INTERVAL, DEVICE,
                        DROPOUT, EVAL_ITERS, LEARNING_RATE, N_EMBD, N_HEADS,
                        N_ITER_TRAIN, N_LAYER, VERB)
from sub.model import (Block, FeedForward, GPTConfig, Head, LayerNorm,
                       MultiHeadAttention)
from sub.server_config import MAX_TRIES

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
    - All nodes run the webserver - it is the key of the application;
        - It also instantiates the corresponding models, and it is used to
        exchange any type of information (config and inference)
    - Main node instantiates GPTDistributed object, which in turn instantiates
    the GPTServer on the same device
    - Main node (through GPTDistributed) sends HTTP messages about their role to 
    each other node in the network (role + weights + model_config + next_node +
    predecessor)
    - The other nodes instantiate the corresponding object (Intermediate/
    Finisher) and open sockets to/from the corresponding nodes, then start 
    waiting for the information to arrive
"""

N_LAYERS_INTERM = 2  # Number of transformer layers in each intermediate node
N_LAYERS_FINISH = 2  # Number of transformer layers in the finisher node
# CTX = (
#     nullcontext()
#     if DEVICE in {"cpu", "mps"}
#     else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# )

script_dir = os.path.dirname(__file__)


def remove_prefix(text: str, prefix: str) -> str:
    """
    Remove the specified prefix from the given string.
    NOTE: starting Python 3.9, use text.removeprefix(prefix)
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

    def load_weights(self, params: Mapping[str, Any]) -> int:
        """Load weights"""
        self.load_state_dict(params)
        self.params_init = True
        if VERB:
            print(f"Weights loaded! Moving model to {self.config.device}")
        self.to(self.config.device)
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

    exposed = True
    model: Union[None, StarterNode, IntermediateNode, FinisherNode] = None
    next_node: Union[None, Dict] = None
    prev_node: Union[None, Dict] = None
    model_params: Union[None, Dict] = None
    model_config: Union[None, GPTConfig] = None
    # NOTE: True iff the model has been initialized and it is ready to perform
    # inference.
    # When this flag is turned to False (at the end of the generated sequence),
    # the loop in self.start() is interrupted; the only way to set this to False
    # (in non-starter nodes) is to receive the specific PUT HTTP request from
    # the starter, that advertises the conclusion of the generated sequence
    running: bool = False
    sock_to_prev: Union[socket.socket, None] = None
    sock_to_next: Union[socket.socket, None] = None

    def __init__(
        self,
        node_type: str,
        node_config: Dict,
        starter_config: Union[
            Dict, None
        ] = None,  # FIXME: add specs for the right parameters
    ):
        """
        Initialize GPTServer object.

        This object will control a specific model (Starter/Intermediate/
        Finisher), allowing to pass on the information in the chain while
        performing inference.

        Args:
            node_type: string indicating the node type - here it is just enough
                to distinguish between "starter" and non-starter (starter is
                configured already here, while non-starters have to be
                configured with a POST request)
            node_config: node configuration information (from .json file)
            starter_config: extra arguments required for the starter node
                params: model parameters (state dict) for starter node
                model_config: GPTConfig object
                next_node: info about next node
                prev_node: info about previous node
                tok_metadata_path: path of the tokenizer metadata (for
                CharacterTokenizer)
        """
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
        # Extract optional parameters
        if node_type.lower() == "starter":
            # Configuration of starter node
            assert type(starter_config) == dict
            assert (
                len(starter_config.keys()) >= 3
            )  # There should be params, next & prev

            if "next_node" in starter_config:  # FIXME: check this works
                self.next_node = dict(starter_config["next_node"])
            else:
                raise ValueError("Missing 'next_node' information")

            if "prev_node" in starter_config:  # FIXME: check this works
                self.prev_node = dict(starter_config["prev_node"])
            else:
                raise ValueError("Missing 'prev_node' information")

            if "params" in starter_config:  # FIXME: check this works
                self.model_params = dict(starter_config["params"])
            else:
                raise ValueError("Missing parameters!")

            if "model_config" in starter_config:
                # Create object as done in train/sample scripts
                assert type(starter_config["model_config"]) == GPTConfig
                self.model_config = starter_config["model_config"]
            else:
                raise ValueError("Missing model configuration")

            if "tok_metadata_path" in starter_config:
                self.tok_meta_path = str(starter_config["tok_metadata_path"])
            else:
                raise ValueError("Missing tokenizer metadata")

            self.init_model(set_eval=True)

        else:
            # Configuration of "secondary" (intermediate or finisher) node
            self.starter_addr = node_config["communication"]["starter_addr"]
            # NOTE: the model will be initialized once config info is received
        pass

    # ----- Public ------------------------------------------------------------

    def init_model(self, set_eval: bool = True):
        """
        Initialize the node's model chunk, passing the parameters

        Args:
            set_eval: if set to true, the model will be set to "eval" mode, used
                to perform inference
        """
        assert self.model_params is not None, "No model parameters were found!"
        assert (
            self.model_config is not None
        ), "No model configuration was found!"
        assert self.model is None, "The model was already initialized!"

        if self.node_type == "starter":
            self.model = StarterNode(self.model_config)
        elif self.node_type == "intermediate":
            self.model = IntermediateNode(self.model_config)
        elif self.node_type == "finisher":
            self.model = FinisherNode(self.model_config)
        else:
            raise ValueError(f"Unsupported node type {self.node_type}")

        self.model.load_weights(self.model_params)
        if set_eval:
            self.model.eval()
        self.running = True  # NOTE!!!

    def start(
        self,
        n_nodes: Union[None, int] = None,
        max_new_tokens: Union[None, int] = None,
    ):
        """
        Perform normal operation (open sockets, wait for communication from
        previous node and forward activations to next one)

        This function launches an infinite loop, interrupted by the receival of
        a special message (PUT) over the communication channel that triggers a
        change in a class attribute

        Args:
            n_nodes: number of nodes in the network, it is the same as the
                number of generated samples (recurrent pipelining)
            max_new_tokens: ONLY FOR STARTER - maximum number of tokens per
                generated sample

            REMOVED: num_samples: ONLY FOR STARTER - number of generated samples
                -> It is the same as the number of nodes in the network
        """
        assert self.running

        # TODO

        n_nodes = 1  # TODO: remove when pipelining is implemented

        if self.node_type == "starter":
            # Open sockets (2 ports!) - starter is server for both prev & next
            assert n_nodes is not None and max_new_tokens is not None
            assert self.sock_to_prev is None and self.sock_to_next is None
            assert self.next_node is not None and self.prev_node is not None
            assert self.model_config is not None and self.model is not None

            if VERB:
                print("Opening socket to next node")
            self.sock_to_next = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            self.sock_to_next.bind(
                (
                    self.node_config["addr"],
                    self.node_config["inference"]["port_out"],
                )
            )
            self.sock_to_next.listen(1)
            self.next_host, self.next_port = self.sock_to_next.accept()
            assert self.next_port == self.next_node["inference"]["port_in"]
            if VERB:
                print(
                    f"Connected to NEXT node: {self.next_host}:{self.next_port}"
                )

            if VERB:
                print("Opening socket to previous node")
            self.sock_to_prev = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            self.sock_to_prev.bind(
                (
                    self.node_config["addr"],
                    self.node_config["inference"]["port_in"],
                )
            )
            self.sock_to_prev.listen(1)
            self.prev_host, self.prev_port = self.sock_to_prev.accept()
            assert self.prev_port == self.prev_node["inference"]["port_out"]
            if VERB:
                print(
                    f"Connected to PREV node: {self.prev_host}:{self.prev_port}"
                )

            # Perform tokenization + encoding -> send to next
            # TOKENIZATION - load metadata of tokenizer (if found)
            # TODO: understand whether to put this here or in GPTDistributed
            if VERB:
                print(f"Loading tokenizer metadata from {self.tok_meta_path}")
            with open(self.tok_meta_path, "rb") as f:
                meta = pickle.load(f)
            self.tok = CharacterTokenizer(meta["stoi"], meta["itos"])
            self.tok_encode = lambda s: self.tok.encode(s)
            self.tok_decode = lambda l: self.tok.decode(l)

            # GENERATION
            # Encode starting sequence (TODO: implement prompt support)
            # FIXME: make idx a vector with n_nodes elements (pipelining)
            start = "\n"
            start_ids = self.tok_encode(start)
            idx = torch.tensor(
                start_ids, dtype=torch.long, device=self.model_config.device
            )[None, ...]

            # TODO: implement recurrent pipelining
            """Hint: use k % n_nodes to decide what to do (on which sample to
            work)
            The idea is: if at iter 'k' we work on a sample, then at iter 'k-1'
            we receive the previous outputs from the finisher
            """
            with torch.no_grad():
                # with CTX:
                for k in range(max_new_tokens * n_nodes):
                    sample_id = k % n_nodes
                    if k >= n_nodes:
                        # We are not in the first iteration (k starts from 0)
                        # Wait for output of corresp. sample from finisher
                        outs = self.sock_to_prev.recv(4)  # TODO: Set bufsize
                        out_logits = pickle.loads(outs)
                        probs = F.softmax(out_logits, dim=1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)

                    # Send to next
                    idx_cond = self.model(idx)

                    out_tx = pickle.dumps(idx_cond)
                    self.sock_to_next.sendall(out_tx)

                print(self.tok_decode(idx[0].tolist()))

        else:
            # TODO: socket creation
            # Intermediate is client for prev node, server for next
            # Finisher is client for prev and next
            pass

    # ----- REST --------------------------------------------------------------

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
        if self.node_type != "starter" and self.model is None:
            init_msg = json.loads(cp.request.body.read())
            self.node_type = init_msg["role"]
            self.prev_node = init_msg["prev_node"]
            self.next_node = init_msg["next_node"]
            self.model_config = init_msg["model_config"]
            self.model_params = init_msg["params"]
            # Set up the node
            self.init_model()
        elif self.model is None:
            raise cp.HTTPError(
                403,
                "Failed to configure node - the model was already initialized",
            )

    def PUT(self):
        """Not implemented"""
        # TODO: for non-Starters, implement stopping condition (use attribute
        # self.running)
        raise cp.HTTPError(501, "PUT not implemented!")

    def DELETE(self):
        """Not implemented"""
        raise cp.HTTPError(501, "PUT not implemented!")


class GPTDistributed:
    """
    Distributed implementation of a minimal GPT2 instance
    """

    # Syntax of the message used to initialize other nodes
    init_msg = {
        "role": "",  # Role name
        "params": {},  # State dict
        "model_config": GPTConfig(),
        "prev_node": {},  # From .json
        "next_node": {},  # From .json
    }

    def __init__(
        self,
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

        if VERB:
            print("Nodes information:\n", json.dumps(self.nodes_info))

        # Store important parameters:
        self.own_config = self.nodes_info["nodes"]["starter"]
        self.own_addr = self.own_config["addr"]
        self.own_comm_port = self.own_config["communication"]["port"]
        self.own_inference_port_in = self.own_config["inference"]["port_in"]
        self.own_inference_port_out = self.own_config["inference"]["port_out"]

        # Get the model parameters and split them based on n. of nodes
        self.n_intermediate = len(self.nodes_info["nodes"]["intermediate"])
        self.n_total_nodes = 2 + self.n_intermediate
        try:
            self.model_ckpt = torch.load(ckpt_path, map_location=DEVICE)
        except:
            if VERB:
                print("Loading full model on RAM - not enough VRAM")
            # It may be that the model does not fit all in the VRAM
            self.model_ckpt = torch.load(ckpt_path, map_location="cpu")

        # Extract state dict & remove problematic keys
        self.complete_model = self.model_ckpt["model"]  # State dict
        unwanted_prefix = "_orig_mod."  # NOTE: this shouldn't happen anymore
        for k, _ in list(self.complete_model.items()):
            if k.startswith(unwanted_prefix):
                self.complete_model[
                    k[len(unwanted_prefix) :]
                ] = self.complete_model.pop(k)
        # Split model
        self.model_chunks = split_parameters(
            model_params=self.complete_model, n_nodes=self.n_total_nodes
        )

        # Extract tokenizer metadata information and check it exists
        if (
            "config" in self.model_ckpt
            and "DATASET" in self.model_ckpt["config"]
        ):
            self.tok_meta_path = os.path.join(
                script_dir,
                "..",
                "data",
                self.model_ckpt["config"]["DATASET"],
                # "shakespeare",
                "meta.pkl",
            )
            assert os.path.exists(
                self.tok_meta_path
            ), f"Unable to find tokenizer data at {self.tok_meta_path}"
        else:
            raise FileNotFoundError("Unable to retrieve tokenizer metadata!")

        self.model_config = GPTConfig(**self.model_ckpt["model_args"])

        # Create webserver
        starter_config = self.init_msg.copy()
        starter_config["role"] = "starter"
        starter_config["params"] = self.model_chunks["starter"]
        starter_config["model_config"] = self.model_config
        starter_config["prev_node"] = self.nodes_info["nodes"]["finisher"]
        if self.n_intermediate:
            starter_config["next_node"] = self.nodes_info["nodes"][
                "intermediate"
            ][0]
        else:
            starter_config["next_node"] = starter_config["prev_node"]

        starter_config["tok_metadata_path"] = self.tok_meta_path

        self.webserv = GPTServer(
            node_type="starter",
            node_config=self.own_config,
            starter_config=starter_config,
        )

    def configure_nodes(self):
        """
        Send POST requests to the other nodes to inform them of their role and
        including their chunk of model.
        """
        # TODO: add also communication information to prev/next node information
        # This way it is possible to ping the nodes (check alive)

        # Store the prev and next in a smart way
        prev = self.nodes_info["nodes"]["starter"]
        n_interm = len(self.nodes_info["nodes"]["intermediate"])
        if n_interm == 0:
            next = prev
        elif n_interm == 1:
            next = self.nodes_info["nodes"]["finisher"]
        else:
            next = self.nodes_info["nodes"]["intermediate"][1]

        # Intermediate config
        for i, int_node in enumerate(self.nodes_info["nodes"]["intermediate"]):
            curr_msg = self.init_msg.copy()
            curr_msg["role"] = "intermediate"
            curr_msg["model_config"] = self.model_config
            curr_msg["params"] = self.model_chunks["intermediate"][i]

            curr_msg["prev_node"] = prev
            curr_msg["next_node"] = next

            # Update next and prev for next iteration
            prev = int_node
            if i == n_interm - 1:  # Last iter in loop
                next = self.nodes_info["nodes"]["starter"]
            elif i == n_interm - 2:  # Second to last iter
                next = self.nodes_info["nodes"]["finisher"]
            else:
                next = self.nodes_info["nodes"]["intermediate"][i + 2]

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
        curr_msg["model_config"] = self.model_config

        curr_msg["prev_node"] = prev
        curr_msg["next_node"] = next

        target_addr = self.nodes_info["nodes"]["finisher"]["addr"]
        target_port = self.nodes_info["nodes"]["finisher"]["communication"][
            "port"
        ]
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
