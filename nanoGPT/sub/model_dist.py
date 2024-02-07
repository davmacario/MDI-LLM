#!/usr/bin/env python3

import inspect
import json
import os
import pickle
import socket
import threading
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple, Union

import cherrypy as cp
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.char_tokenizer import CharacterTokenizer
from sub.config import (BATCH_SIZE, BIAS, BLOCK_SIZE, CKPT_INTERVAL, DEVICE,
                        DROPOUT, EVAL_ITERS, LEARNING_RATE, N_EMBD, N_HEADS,
                        N_ITER_TRAIN, N_LAYER, TEMPERATURE, TOP_K, VERB)
from sub.model import (Block, FeedForward, GPTConfig, Head, LayerNorm,
                       MultiHeadAttention)
from sub.server_config import MAX_TRIES
from sub.utils import loading_bar

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

N_LAYERS_INTERM = 1  # Number of transformer layers in each intermediate node
N_LAYERS_FINISH = 1  # Number of transformer layers in the finisher node
# CTX = (
#     nullcontext()
#     if DEVICE in {"cpu", "mps"}
#     else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# )
HEADERLENGTH = 10  # Header length in chars

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

    out_chunks = {}

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
                    end = remove_prefix(k, prefix)
                    new_k = f"intermediate_model.layers.{local_layer_ind}{end}"
                    curr_params[new_k] = model_params[k]
            local_layer_ind += 1

        out_chunks["intermediate"].append(curr_params)

    # 3. Select params for Finisher
    out_chunks["finisher"] = {}

    # Layers:
    valid_layer_ind = list(
        range((n_nodes - 2) * N_LAYERS_FINISH, (n_nodes - 1) * N_LAYERS_FINISH)
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
                end = remove_prefix(k, prefix)
                new_k = f"finisher_model.layers.{local_layer_ind}{end}"
                out_chunks["finisher"][new_k] = model_params[k]
        local_layer_ind += 1
    # OK below
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
        out_chunks["finisher"][f"finisher_model.lm_head.bias"] = model_params[
            f"{output_layer}.bias"
        ]
    except:
        pass

    return out_chunks


def serialize_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Serialize a mapping, specifically a state dict, to allow it to be read as a
    JSON/dict.
    """
    json_serializable_params = {}
    for key, value in params.items():
        json_serializable_params[key] = (
            value.tolist() if isinstance(value, torch.Tensor) else value
        )

    return json_serializable_params


def deserialize_params(params: Dict) -> Mapping[str, Any]:
    """
    De-serialize a dictionary and return a state dictionary containing a torch
    model parameters.
    """
    deserialized_params = {}
    for key, value in params.items():
        if isinstance(value, list):
            # Convert lists back to PyTorch tensors
            deserialized_params[key] = torch.tensor(value)
        else:
            deserialized_params[key] = value

    return deserialized_params


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
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
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
    model: Union[StarterNode, IntermediateNode, FinisherNode, None] = None
    next_node: Union[Dict, None] = None
    prev_node: Union[Dict, None] = None
    model_params: Union[Dict, None] = None
    model_config: Union[GPTConfig, None] = None
    # True iff the model has been initialized and it is ready to perform
    # inference.
    running: bool = False
    stop_msg: str = "STOP"
    sock_to_prev: Union[socket.socket, None] = None
    sock_to_prev_prop: Tuple = ()  # FIXME - not needed
    sock_to_next: Union[socket.socket, None] = None
    sock_to_next_prop: Tuple = ()

    # Some model configs:
    top_k = TOP_K
    temperature = TEMPERATURE

    def __init__(
        self,
        node_config: Dict,
        node_type: Union[None, str] = None,
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

        self.node_type = node_type
        self.node_config = node_config
        # Extract optional parameters
        if node_type is not None and node_type.lower() == "starter":
            # Configuration of starter node
            assert type(starter_config) == dict
            assert (
                len(starter_config.keys()) >= 3
            )  # There should be params, next & prev

            if "next_node" in starter_config:
                self.next_node = dict(starter_config["next_node"])
            else:
                raise ValueError("Missing 'next_node' information")

            if "prev_node" in starter_config:
                self.prev_node = dict(starter_config["prev_node"])
            else:
                raise ValueError("Missing 'prev_node' information")

            if "params" in starter_config:
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
            self._running_thread = threading.Thread()  # Placeholder
            # NOTE: the model will be initialized once config info is received (POST)

        self.webserv_config = {
            "/": {
                "request.dispatch": cp.dispatch.MethodDispatcher(),
                "tools.sessions.on": True,
            }
        }

        cp.tree.mount(self, "/", self.webserv_config)
        cp.config.update(
            {
                "server.socket_host": self.own_addr,
                "server.socket_port": self.own_comm_port,
            }
        )

        cp.engine.start()

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
        self.model = self.model.to(DEVICE)
        if set_eval:
            self.model.eval()

    def start(
        self,
        n_nodes: Union[None, int] = None,
        max_new_tokens: Union[None, int] = None,
    ) -> Union[None, str]:
        """
        Perform normal operation (open sockets, wait for communication from
        previous node and forward activations to next one)

        In starter nodes, the function launches the operation by creating
        sockets to the nodes and initializing the sample vectors.
        Starter nodes are the only ones for which the arguments should not be
        None.
        The loop, for starter nodes, is not infinite, as they should know how
        many tokens to generate.

        This function launches an infinite loop on a separate thread in
        non-starter nodes, interrupted by the receival of a special message
        (PUT) over the communication channel that triggers a change in a class
        attribute.
        Non-starter node do not know how long the generation will take, hence
        they need to be stopped "externally" by the starter node once the
        generation is complete.

        Args:
            n_nodes: number of nodes in the network, it is the same as the
                number of generated samples (recurrent pipelining)
            max_new_tokens: ONLY FOR STARTER - maximum number of tokens per
                generated sample
        """
        n_nodes = 1  # TODO: remove when pipelining is implemented

        # Configuration for all nodes
        assert self.sock_to_prev is None and self.sock_to_next is None
        assert self.next_node is not None and self.prev_node is not None
        assert self.model_config is not None and self.model is not None

        self.create_sockets()

        assert self.sock_to_prev is not None and self.sock_to_next is not None

        # Differentiate between different types
        if self.node_type == "starter":
            assert n_nodes is not None and max_new_tokens is not None

            self._load_tokenizer()
            self.tok_encode = lambda s: self.tok.encode(s)
            self.tok_decode = lambda l: self.tok.decode(l)

            if VERB:
                print("> Tokenizer loaded!")

            # GENERATION
            # Encode starting sequence (TODO: implement prompt support)

            # TODO: implement recurrent pipelining
            """Hint: use k % n_nodes to decide what to do (on which sample to
            work)
            The idea is: if at iter 'k' we work on a sample, then at iter 'k-1'
            we receive the previous outputs from the finisher
            """
            # FIXME: make idx a vector with n_nodes elements (pipelining)
            # idx will contain all the different n_nodes samples
            start = "\n"
            start_ids = self.tok_encode(start)
            idx = torch.tensor(
                start_ids, dtype=torch.long, device=self.model_config.device
            )[None, ...]

            # TODO: implement mechanism to stop generation whenever the token
            # corresp to EOS is received for one of the samples
            with torch.no_grad():
                # with CTX:  # FIXME
                total_iters = max_new_tokens * n_nodes
                for k in range(total_iters):
                    sample_id = k % n_nodes
                    print(
                        f"Generating: {loading_bar(k, total_iters, 20)}",
                        end="\r",
                    )
                    if k >= n_nodes:
                        # We are not in the first iteration (k starts from 0)
                        # Wait for output of corresp. sample from finisher
                        out_logits = self.recv_from_prev().to(
                            self.model_config.device
                        )
                        logits = out_logits[:, -1, :] / self.temperature
                        if self.top_k is not None:
                            v, _ = torch.topk(
                                logits, min(self.top_k, logits.size(-1))
                            )
                            logits[logits < v[:, [-1]]] = -float("Inf")
                        probs = F.softmax(logits, dim=1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)

                    if k < (n_nodes * (max_new_tokens - 1)):
                        # Send to next iff not at the last token

                        # Crop to block size
                        idx_cond = (
                            idx
                            if idx.size(1) <= self.model.config.block_size
                            else idx[:, -self.model.config.block_size :]
                        )
                        # Forward in local model
                        idx_cond = self.model(idx_cond)

                        self.send_to_next(idx_cond)

                print("Generation completed!                          ")

                out_text = self.tok_decode(idx[0].tolist())
                # Stop timer

                # Send stop message to the next
                self.send_to_next(self.stop_msg)

                return out_text
        else:
            self.running = True
            self._node_loop()

    def recv_from_prev(self, max_wait_s: int = 5) -> Any:
        """
        Receive a full message from the previous node.

        This method assumes the message is an encoding in bytes of any Python
        object, which can be extracted using pickle.

        Returns:
            the received python object
        """
        assert self.sock_to_prev is not None

        full_msg = b""
        new_msg = True
        finished = False
        t_start = time.time()
        while not finished and time.time() - t_start < max_wait_s:
            # Receive information from the new socket
            msg = self.sock_to_prev.recv(1024)  # Size of the buffer in bytes
            if new_msg:
                # Extract message length from the header
                msg_len = int(msg[:HEADERLENGTH])
                new_msg = False

            full_msg += msg

            if len(full_msg) - HEADERLENGTH >= msg_len:
                # When the full message is received, the length will be the one in the header
                # print("Raw msg: ", full_msg[HEADERLENGTH:])
                data = pickle.loads(full_msg[HEADERLENGTH:])
                # Look for stopping msg
                if type(data) == str and data == self.stop_msg:
                    if VERB:
                        print("Stopping message received! Generation complete!")
                    self.running = False
                    return self.stop_msg
                else:
                    finished = True
                    return data

    def send_to_next(self, data: Any):
        """
        Send any Python object to the next node.
        """
        assert self.sock_to_next_prop != ()

        message_str = pickle.dumps(data)
        tx_msg = (
            bytes(f"{len(message_str):<{HEADERLENGTH}}", "utf-8") + message_str
        )
        self.sock_to_next_prop[0].send(tx_msg)

    def create_sockets(self):
        """
        Create sockets for communicating the intermediate results with the
        previous and next nodes in the chain.

        Starter nodes will open the connection towards the next node first,
        while all other nodes will first connect to the previous ones.
        """
        assert self.sock_to_prev is None and self.sock_to_next is None
        assert self.next_node is not None and self.prev_node is not None

        if self.node_type == "starter":
            # Open server towards next node (first thing if starter node)
            if VERB:
                print(
                    f"Opening socket to next node (to port {self.next_node['inference']['port_in']})"
                )

            self._start_server()
            assert self.sock_to_next is not None
            self.sock_to_next.listen(1)

            self.sock_to_next_prop = self.sock_to_next.accept()

            if VERB:
                print(f"Connected to NEXT node: {self.sock_to_next_prop[1]}")

        # Open client towards previous
        if VERB:
            print(
                f"Opening socket to previous node (to port {self.prev_node['inference']['port_out']})"
            )

        self._start_client()
        assert self.sock_to_prev is not None

        if VERB:
            print("Connected to PREV node as client")
        self.running = True

        if self.node_type != "starter":
            # Open server towards next node
            if VERB:
                print(
                    f"Opening socket to next node (to port {self.next_node['inference']['port_in']})"
                )

            self._start_server()
            assert self.sock_to_next is not None
            self.sock_to_next.listen(1)

            self.sock_to_next_prop = self.sock_to_next.accept()

            if VERB:
                print(f"Connected to NEXT node: {self.sock_to_next_prop[1]}")

    def shutdown(self) -> int:
        """
        Turn off the node - stop server, close sockets and stop thread.

        Returns:
            1 upon success, 0 otherwise
        """
        try:
            time.sleep(3)
            cp.engine.stop()
            self.sock_to_prev.close()  # FIXME: one of the 2 should use the conn.
            self.sock_to_next.close()
            self.running = False  # Redundant
            if self.node_type != "starter":
                self._running_thread.join()
            return 1
        except:
            return 0

    # ----- Private -----------------------------------------------------------

    def _load_tokenizer(self) -> CharacterTokenizer:
        """
        Load the tokenizer information from the path specified in class
        attribute `self.tok_meta_path`.

        The tokenizer object will be stored in `self.tok`.

        Returns:
            the tokenizer object
        """
        if VERB:
            print(f"Loading tokenizer metadata from {self.tok_meta_path}")
        with open(self.tok_meta_path, "rb") as f:
            meta = pickle.load(f)
        self.tok = CharacterTokenizer(meta["stoi"], meta["itos"])

        return self.tok

    def _start_server(self, max_tries: int = 30):
        """
        Start the server socket, i.e., the socket to the next node in the chain.
        """
        loopsigns = ["|", "/", "-", "\\"]
        self.sock_to_next = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        failed = True
        tries = 0
        while failed and tries < max_tries:
            # Attempt to bind
            try:
                self.sock_to_next.bind(
                    (
                        self.node_config["addr"],
                        self.node_config["inference"]["port_out"],
                    )
                )
            except:
                tries += 1
                print(f"Retrying {loopsigns[tries % 4]}", end="\r")
                time.sleep(1)
            else:
                failed = False

        if failed:
            raise ConnectionError(
                f"Unable to bind to ({self.node_config['addr']}, {self.node_config['inference']['port_out']})"
            )
        # Will listen and accept afterwards

    def _start_client(self, max_tries: int = 30):
        """
        Start the client socket, i.e., the socket to the prev node in the chain.
        """
        loopsigns = ["|", "/", "-", "\\"]
        conn = False
        tries = 0
        while not conn and tries < max_tries:
            try:
                self.sock_to_prev = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM
                )
                # Bind should work even after some fails
                self.sock_to_prev.bind(
                    (
                        self.node_config["addr"],
                        self.node_config["inference"]["port_in"],
                    )
                )
                self.sock_to_prev.connect(
                    (
                        self.prev_node["addr"],
                        self.prev_node["inference"]["port_out"],
                    )
                )
            except:
                # Can either fail when binding or when connecting
                tries += 1
                print(f"Retrying {loopsigns[tries % 4]}", end="\r")
                time.sleep(1)
            else:
                conn = True

        if not conn:
            raise ConnectionError(
                f"Unable to create client socket at ({self.node_config['addr']}, {self.node_config['inference']['port_in']})"
            )

    def _node_loop(self):
        """
        Execution loop for non-starter nodes. This method must be used as the
        target of a thread that is launched once the node has been correctly
        initialized.

        The execution will be stopped once a PUT request is made to /stop.
        """
        assert self.sock_to_prev is not None and self.sock_to_next is not None
        assert self.model is not None and self.model_config is not None

        loopsigns = ["|", "/", "-", "\\"]
        iter = 0
        with torch.no_grad():
            while self.running:
                # Wait for output from prev or stopping condition
                ins = self.recv_from_prev()
                if self.running:
                    print(f"Generating {loopsigns[iter % 4]}", end="\r")
                    ins = ins.to(self.model_config.device)
                    # Forward pass
                    outs = self.model(ins)
                    # Send to next
                    self.send_to_next(outs)
                    iter += 1
                else:
                    print("Generation completed!")
                    self.send_to_next(self.stop_msg)

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
        if self.node_type is None and self.model is None:
            if len(path) > 0 and path[0] == "init":
                assert not self.running
                init_msg = json.loads(cp.request.body.read())
                self.node_type = init_msg["role"]
                self.prev_node = init_msg["prev_node"]
                self.next_node = init_msg["next_node"]
                self.model_config = GPTConfig(**init_msg["model_config"])
                self.model_params = deserialize_params(init_msg["params"])
                # Set up the node
                self.init_model()
                if VERB:
                    print(f"Starting operation - {self.node_type} node")
                self._running_thread = threading.Thread(
                    target=self.start, daemon=True
                )
                self._running_thread.start()
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found")
        elif self.model is None:
            raise cp.HTTPError(
                403,
                "Failed to configure node - the model was already initialized",
            )

    def PUT(self, *path):
        """
        Used by the starter to stop running nodes at the end of the generation
        """
        if self.node_type not in {"intermediate", "finisher"}:
            raise cp.HTTPError(501, "PUT not implemented!")
        else:
            if len(path) > 0 and path[0] == "stop":
                self._end_thr = threading.Thread(
                    target=self.shutdown, daemon=True
                )
                self._end_thr.start()
                if VERB:
                    print("Node stopped!")
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found!")

    def DELETE(self):
        """Not implemented"""
        raise cp.HTTPError(501, "DELETE not implemented!")


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

    def configure_nodes(self) -> int:
        """
        Send POST requests to the other nodes to inform them of their role and
        including their chunk of model.

        Information sent:
            - Node role ("role")
            - Model config (GPTConfig as dict) ("model_config")
            - Model parameters ("params") - from pickle.dumps()
            - Previous node information - from json file ("prev_node")
            - Next node information - from json ("next_node")

        Returns:
            1 if success
            0 if at least 1 node fails
        """
        out = 1  # Return code

        # Store the prev and next in a smart way
        prev = self.nodes_info["nodes"]["starter"]
        n_interm = len(self.nodes_info["nodes"]["intermediate"])
        if n_interm == 0:  # No intermediate nodes
            next = prev
        elif n_interm == 1:
            next = self.nodes_info["nodes"]["finisher"]
        elif n_interm > 1:
            next = self.nodes_info["nodes"]["intermediate"][1]
        else:
            raise ValueError("Should not be here!")

        # Intermediate config
        for i, int_node in enumerate(self.nodes_info["nodes"]["intermediate"]):
            if VERB:
                print(f"Initializing intermediate node n.{i}")
            curr_msg = self.init_msg.copy()
            curr_msg["role"] = "intermediate"
            curr_msg["model_config"] = self.model_config.asdict()
            curr_msg["params"] = serialize_params(
                self.model_chunks["intermediate"][i]
            )

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
            out *= self.request_to_node("post", addr, curr_msg)

            if not out:
                if VERB:
                    print("> Failed!")
                return out

            if VERB:
                print("> Success!")

        # Finisher config - can use next/prev from last loop iteration
        curr_msg = self.init_msg.copy()
        curr_msg["role"] = "finisher"
        curr_msg["model_config"] = self.model_config.asdict()
        curr_msg["params"] = serialize_params(self.model_chunks["finisher"])

        curr_msg["prev_node"] = prev
        curr_msg["next_node"] = next

        target_addr = self.nodes_info["nodes"]["finisher"]["addr"]
        target_port = self.nodes_info["nodes"]["finisher"]["communication"][
            "port"
        ]
        addr = f"http://{target_addr}:{target_port}/init"
        if VERB:
            print(f"Initializing finisher node ({addr})")

        out *= self.request_to_node("post", addr, curr_msg)

        if VERB and out:
            print("> Success!")
        elif VERB and not out:
            print("> Failed!")

        return out

    def request_to_node(
        self, req_type: str, addr: str, content: dict, max_n_requests: int = 100
    ) -> int:
        """
        Send an HTTP request containing a json-formatted string to a specified
        target node.

        Args:
            req_type: type of HTTP request, can be "post" or "put"
            addr: full address (http(s)://<ip>:<port>) of the target node
            content: python dict containing the information
            max_n_requests: maximum number of requests before failure

        Returns:
            1 if successful
            0 if failed
        """
        if req_type.lower() == "post":
            req_func = requests.post
        elif req_type.lower() == "put":
            req_func = requests.put
        else:
            raise ValueError(f"Unsupported request type '{req_type}'")
        ret = None
        n_ret = 0
        try:
            ret = req_func(addr, json=content)
        except:
            n_ret += 1
        while (
            ret is None or ret.status_code != 200
        ) and n_ret < max_n_requests:
            if VERB:
                print(
                    f"Unable to reach node - retrying in 3s ({n_ret}/{max_n_requests})"
                )
            time.sleep(3)
            try:
                ret = req_func(addr, json=content)
            except:
                n_ret += 1

        if ret is not None and ret.status_code == 200:
            return 1
        return 0

    def start(self):
        """
        Start the operation - webserver + model

        Stop when the model finished running, i.e., all tokens have been
        generated.

        This method calls back self.configure_nodes() and self.webserv.start()
        """
        # TODO: add assertions (uninitialized values)
        if not self.configure_nodes():
            raise AssertionError("Unable to initialize required nodes!")

        out = self.webserv.start(
            n_nodes=self.n_total_nodes, max_new_tokens=1000
        )

        print("-------------------------------------------------")
        print("Produced output:\n", out)
        print("-------------------------------------------------")

        # Once finished, send PUT to each node to terminate execution for them
        self.stop()

    def stop(self) -> int:
        """
        Interrupt operation and shut down the node.

        Terminate execution of the application on every other node by sending
        them a PUT request to <addr>:<port>/stop.

        Returns:
            1 if all requests were successful
            0 if at least 1 request failed
        """
        out = 1
        for int_node in self.nodes_info["nodes"]["intermediate"]:
            target_addr = int_node["addr"]
            target_port = int_node["communication"]["port"]

            addr = f"http://{target_addr}:{target_port}/stop"
            if VERB:
                print(f"Sending PUT request to {addr}")
            out *= self.request_to_node("PUT", addr, {})

        target_addr = self.nodes_info["nodes"]["finisher"]["addr"]
        target_port = self.nodes_info["nodes"]["finisher"]["communication"][
            "port"
        ]
        addr = f"http://{target_addr}:{target_port}/stop"
        if VERB:
            print(f"Sending PUT request to {addr}")
        out *= self.request_to_node("PUT", addr, {})

        self.webserv.shutdown()

        return out
