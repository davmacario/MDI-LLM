#!/usr/bin/env python3

# Copyright 2024 Davide Macario (@davmacario)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import logging
import os
import pickle
import socket
import threading
import time
from collections import deque
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Tuple, Union

import cherrypy as cp
import requests
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.bpe_tokenizer import BPETokenizer
from sub.char_tokenizer import CharacterTokenizer
from sub.config import (DEVICE, DTYPE, HEADERLENGTH, PLOTS, TEMPERATURE, TOP_K,
                        VERB)
from sub.model import Block, GPTConfig, LayerNorm
from sub.utils import (load_from_hf, loading_bar, plot_tokens_per_time,
                       split_parameters)

"""
Distributed implementation of GPT2 - using the same blocks as the original model.

Rationale:
    - Three block types:
        a. Starter: embedding (token + positional), dropout, transformer layers + 
            final linear layer
        b. Secondary: transformer layers only (generic worker node)
    - The "Starter" is the main node - it will initiate inference and also evaluate the
        final logits (outputs of the model) from the last Secondary node
        - The last layer has been moved to the starter node, and is executed at the end,
        because this allows to transmit the same size of data (i.e., tensors of size
        "N_EMBD" - the length of the embedded data) between all the nodes.
    - Each node opens 2 ports: one for communication (roles setup and weight exchange,
        using HTTP), the other one for the actual inference (i.e., exchange of
        activations)
    - All configuration information is stored in a .json file
        (see src/GPT2/settings_distr)
    - The model parameters can be either split by the starter node itself (if the model
        can fit in its memory completely) or can be loaded from chunks stored on the 
        disk of each device.

Functioning:
    - All nodes run the webserver - it is the key of the application;
        - It also instantiates the corresponding models, and it is used to
        exchange any type of information (config and inference)
    - Main node instantiates GPTDistributed object, which in turn instantiates
    the GPTServer on the same device
    - Main node (through GPTDistributed) sends HTTP messages about their role to each
    other node in the network (role + optional weights + model_config + next_node +
    predecessor)
        - If not sending the weights, the other nodes will load their own model chunk
            from disk
    - The other nodes instantiate the 'SecondaryNode' object and open sockets to/from
        the corresponding nodes, then start waiting for the information to arrive.
    - The main generation loop is ran by the Starter node (others don't know how many
        iterations they will perform, they just know that they will have to pass the 
        incoming message through their local piece of the model and transmit the output
        to the next node in the chain).
    - Upon generation completion, the starter node will send the "finish" message to all
        other nodes to signal the conclusion.

Transmission:
    - Configuration information is sent over HTTP; each node is also an HTTP server.
        - The starter will perform POST requests to assign roles and PUT requests to 
        signal the program termination
    - Transmission of the intermediate activations is done over TCP/IP sockets; this is
    necessary because it lacks the overhead of a fully-fledged application layer
    protocol (such as HTTP) allowing for faster transmission.
        - The message exchange protocol is very simple: each message will have a
        fixed-length header indicating the message size in bytes, and the payload will
        only contain those bytes.
        - This structure allows each node to catch the message length, and to always be
        able to expect the correct amount of bytes.
    - Transmission during inference is handled as follows: each node creates a message
    FIFO queue to store incoming messages on a separate thread from the main one (which
    runs the inference). The main node will then look for incoming messages inside the
    queue.
    This procedure allows to execute the main loop without interruptions and
    independently on each device.
        - The queue is a Python deque object, optimized for 'append' and 'pop'
        operations
"""
# Logging
script_dir = os.path.dirname(__file__)
logger_wp = logging.getLogger("model_dist")
logger_wp.setLevel(logging.ERROR)

MODEL_TYPE = ""
ctx = nullcontext()


class StarterNode(nn.Module):
    """Starter node"""

    params_init = False

    def __init__(self, config: GPTConfig, n_transf_layers: int):
        super().__init__()
        assert config.vocab_size is not None

        self.config = config

        self.starter_model = nn.ModuleDict(
            dict(
                # Initial layers
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(n_transf_layers)]),
                # Final layer:
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
                lm_head=nn.Linear(config.n_embd, config.vocab_size, bias=False),
            )
        )

        # Including the final linear layer in the starter allows for weight-tying
        self.starter_model.wte.weight = self.starter_model.lm_head.weight

    def load_weights(self, params: Mapping[str, Any]) -> int:
        """Load weights"""
        try:
            self.load_state_dict(params)
        except RuntimeError:
            missing_k, unwanted_k = self.load_state_dict(params, strict=False)
            if len(missing_k) > 0:
                raise RuntimeError(
                    f"The model is missing {len(missing_k)} keys:\n\t{missing_k}"
                )
            # Only allow '[].attn.bias' as extra keys - triangular mask
            if not all([k.endswith(".attn.bias") for k in unwanted_k]):
                raise RuntimeError(f"Unrecognized extra keys:\n\t{unwanted_k}")
        self.params_init = True
        if VERB:
            print(f"Weights loaded!")
        logger_wp.info(f"Weights loaded!")
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
        tok_emb = self.starter_model.wte(idx)

        # Obtain positional embeddings by encoding values (0, ..., t)
        pos_emb = self.starter_model.wpe(torch.arange(t, device=device))

        x = self.starter_model.drop(tok_emb + pos_emb)  # (B, T, C)

        for block in self.starter_model.h:
            x = block(x)

        return x

    def forward_last(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Last forward pass - starter node.
        """
        idx = self.starter_model.ln_f(idx)
        return self.starter_model.lm_head(idx)


class SecondaryNode(nn.Module):
    """Secondary worker node"""

    params_init = False

    def __init__(self, config: GPTConfig, n_transf_layers: int):
        super().__init__()
        assert config.vocab_size is not None

        self.config = config

        # Follow naming convention
        self.secondary_model = nn.ModuleDict(
            dict(h=nn.ModuleList([Block(config) for _ in range(n_transf_layers)]))
        )

    def load_weights(self, params: Mapping[str, Any]) -> int:
        """Load weights"""
        try:
            self.load_state_dict(params)
        except RuntimeError:
            missing_k, unwanted_k = self.load_state_dict(params, strict=False)
            if len(missing_k) > 0:
                raise RuntimeError(
                    f"The model is missing {len(missing_k)} keys:\n\t{missing_k}"
                )
            if not all([k.endswith(".attn.bias") for k in unwanted_k]):
                raise RuntimeError(f"Unrecognized extra keys:\n\t{unwanted_k}")
        self.params_init = True
        if VERB:
            print(f"Weights loaded!")
        return 1

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass - secondary node"""
        if not self.params_init:
            raise ValueError("The model parameters have not been initialized!")
        x = idx
        for block in self.secondary_model.h:
            x = block(x)
        return x


# -----------------------------------------------------------------------------


class GPTServer:
    """
    Communication server - Cherrypy-based webserver used for exchanging
    (receiving) setup and control information
    """

    exposed = True
    model: Union[StarterNode, SecondaryNode, None] = None
    next_node: Union[Dict, None] = None
    prev_node: Union[Dict, None] = None
    model_params: Union[Dict, None] = None
    model_config: Union[GPTConfig, None] = None
    # True iff the model has been initialized and it is ready to perform
    # inference.
    running: bool = False
    stop_msg = {"stop": True}
    sock_to_prev: Union[socket.socket, None] = None
    sock_to_prev_prop: Tuple = ()  # NOTE: used now
    sock_to_next: Union[socket.socket, None] = None
    sock_to_next_prop: Tuple = ()  # NOTE: not used!

    # Message queue
    message_queue = deque([])

    # Some model configs:
    top_k = TOP_K
    temperature = TEMPERATURE

    # Stats - n. tokens/time (tuples)
    tok_time = []

    def __init__(
        self,
        node_config: Dict,
        node_type: Union[None, str] = None,
        starter_config: Union[Dict, None] = None,
        chunk_path: Union[None, str] = None,
        **kwargs,
    ):
        """
        Initialize GPTServer object.

        This object will control a specific model (Starter/Secondary), allowing to pass
        on the information in the chain while performing inference.

        Args:
            node_type: string indicating the node type - here it is just enough to
                distinguish between "starter" and non-starter (starter is configured
                already here, while non-starters have to be configured with a POST
                request)
            node_config: node configuration information (from .json file)
            starter_config: extra arguments required for the starter node; expected
                keys:
                - params: model parameters (state dict) for starter node
                - model_config: GPTConfig object
                - next_node: info about next node
                - prev_node: info about previous node
                - tok_metadata_path: path of the tokenizer metadata (for
                    CharacterTokenizer)
            chunk_path: if SecondaryNode, optional path of the model chunk to be used.
                If not set, the program will expect the chunk from the starter node.
        """
        self.device = DEVICE

        # Override global constants
        if "verb" in kwargs:
            global VERB
            print("Overriding 'verb': True")
            VERB = bool(kwargs["verb"])
        if "plots" in kwargs:
            global PLOTS
            print("Overriding 'plots': True")
            PLOTS = bool(kwargs["plots"])

        self.own_addr = node_config["addr"]
        self.own_comm_port = node_config["communication"]["port"]

        self.node_type = node_type
        self.node_config = node_config
        # Possibly get device info if found in config file
        self.device = DEVICE if "device" not in node_config else node_config["device"]

        # Extract optional parameters
        if node_type is not None and node_type.lower() == "starter":
            # Configuration of starter node
            assert type(starter_config) == dict
            req_keys = {
                "n_nodes",
                "next_node",
                "prev_node",
                "params",
                "model_config",
                "tok_metadata_path",
            }
            assert all([k in starter_config for k in req_keys])

            self.n_nodes = starter_config["n_nodes"]
            self.next_node = dict(starter_config["next_node"])
            self.prev_node = dict(starter_config["prev_node"])
            self.model_params = dict(starter_config["params"])
            self.model_config = starter_config["model_config"]
            self.tok_meta_path = str(starter_config["tok_metadata_path"])
            self.device = starter_config["device"]

            self.init_model(starter_config["n_layers"])

        else:
            # Configuration of "secondary" node
            self.starter_addr = node_config["communication"]["starter_addr"]
            self._running_thread = threading.Thread()  # Placeholder
            self.chunk_path = chunk_path
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
                "server.thread_pool": 8,
                # remove any limit on the request body size; default is 100MB
                "server.max_request_body_size": 0,
                # increase server socket timeout to 60s; default is 10s
                "server.socket_timeout": 10000,
            }
        )

        cp.engine.start()

    # ----- Public ------------------------------------------------------------

    def init_model(self, n_transf_layers: int):
        """
        Initialize the node's model chunk, passing the parameters.

        The model will also be moved to the target device.

        Args:
            set_eval: if set to true, the model will be set to "eval" mode, used to
                perform inference
        """
        assert self.model_params is not None, "No model parameters were found!"
        assert self.model_config is not None, "No model configuration was found!"
        assert self.model is None, "The model was already initialized!"

        if self.node_type == "starter":
            self.model = StarterNode(self.model_config, n_transf_layers)
        elif self.node_type == "secondary":
            self.model = SecondaryNode(self.model_config, n_transf_layers)
        else:
            raise ValueError(f"Unsupported node type {self.node_type}")

        self.model = self.model.to(self.device)
        self.model.load_weights(self.model_params)

    def start(
        self,
        max_new_tokens: Union[None, int] = None,
    ) -> Union[None, Tuple[List[str], float]]:
        """
        Perform normal operation (open sockets, wait for communication from previous
        node and forward activations to next one)

        In starter nodes, the function launches the operation by creating sockets to the
        nodes and initializing the sample vectors.
        Starter nodes are the only ones for which the arguments should not be None.
        The loop, for starter nodes, is not infinite, as they should know how many
        tokens to generate.

        This function launches an infinite loop on a separate thread in non-starter
        nodes, interrupted by the receival of a special message (PUT) over the
        communication channel that triggers a change in a class attribute.
        Non-starter node do not know how long the generation will take, hence they need
        to be stopped "externally" by the starter node once the generation is complete.

        Args:
            n_nodes: number of nodes in the network, it is the same as the number of
                generated samples (recurrent pipelining)
            max_new_tokens: ONLY FOR STARTER - maximum number of tokens per generated
                sample

        Returns:
            if starter node, return the list of produced samples, else nothing
        """
        assert self.sock_to_prev is None and self.sock_to_next is None
        assert self.next_node is not None and self.prev_node is not None
        assert self.model_config is not None and self.model is not None

        # Configuration for all nodes
        self.create_sockets()

        assert self.sock_to_prev is not None and self.sock_to_next is not None

        # Differentiate between different types
        if self.node_type == "starter":
            assert max_new_tokens is not None

            self._load_tokenizer()
            if isinstance(self.tok, tiktoken.Encoding):
                self.tok_encode = lambda s: self.tok.encode(
                    s, allowed_special={"<|endoftext|>"}
                )
            else:
                self.tok_encode = self.tok.encode
            self.tok_decode = self.tok.decode

            if VERB:
                print("[INFO] Tokenizer loaded!")
                print("[INFO] Starting queue thread")
            logger_wp.info("Tokenizer loaded!")
            logger_wp.info("Starting queue thread")

            self.queue_thread = threading.Thread(target=self._fill_queue, daemon=True)
            self.queue_thread.start()

            if VERB:
                print("[INFO] Starting generation loop")
            logger_wp.info("Starting generation loop")

            out_text, gen_time = self._starter_loop(max_new_tokens)

            return out_text, gen_time
        else:
            self.running = True
            if VERB:
                print("[INFO] Starting queue thread")
            logger_wp.info("Starting queue thread")
            self.queue_thread = threading.Thread(target=self._fill_queue, daemon=True)
            self.queue_thread.start()

            if VERB:
                print("[INFO] Starting generation loop")
            logger_wp.info("Starting generation loop")
            self._node_loop()

    def recv_from_prev(self, size: int) -> bytes:
        """
        Receive a message of the specified size from the previous node.

        Remark: the size specified in socket.recv(<>) is the MAX size that will be read
        from the receiver buffer.

        Args:
            size: size (in bytes) of the expected message

        Returns:
            the received message (NOT decoded)
        """
        assert self.sock_to_prev is not None and self.sock_to_prev_prop != ()

        full_msg = b""
        while self.running and len(full_msg) < size:
            msg = self.sock_to_prev_prop[0].recv(size - len(full_msg))
            if not msg:
                # Prev node shut connection down (error)
                self.running = False
                logger_wp.error("Connection was terminated unexpectedly!")
            full_msg += msg
            if not self.running:
                break
        return full_msg

    def send_to_next(self, data: Any):
        """
        Send any Python object to the next node.
        The sender is a **client**.

        The message is composed by a header of HEADERLENGTH bytes including the length
        of the actual message, plus a message of MSGLENGTH bytes containing the
        zero-padded message.
        """
        assert self.sock_to_next is not None

        message_str = pickle.dumps(data)
        tx_msg = bytes(f"{len(message_str):<{HEADERLENGTH}}", "utf-8") + message_str
        # NOTE: attempt at sending multiple messages in a "safe" way (no sendall)
        while tx_msg:
            tx_msg = tx_msg[self.sock_to_next.send(tx_msg) :]
        logger_wp.debug("Sent full message to next")

    def create_sockets(self):
        """
        Create sockets for communicating the intermediate results with the previous and
        next nodes in the chain.

        Starter nodes will open the connection towards the next node first, while all
        other nodes will first connect to the previous ones (otherwise the application
        would just wait indefinitely, as no node will connect with any other).
        """
        assert self.sock_to_prev is None and self.sock_to_next is None
        assert self.next_node is not None and self.prev_node is not None

        if self.node_type == "starter":
            # Open server towards next node (first thing if starter node)
            if VERB:
                print(
                    f"[INFO] Opening socket to next node (to port {self.next_node['inference']['port_in']})"
                )

            self._start_client()
            assert self.sock_to_next is not None
            logger_wp.info("Created socket to next node")

            if VERB:
                print("-> Done!                     ")

        # Open client towards previous
        if VERB:
            print(
                f"[INFO] Opening socket to previous node (to port {self.prev_node['inference']['port_out']})"
            )

        self._start_server()
        assert self.sock_to_prev is not None
        if VERB:
            print("Started listening")
        self.sock_to_prev.listen(1)

        self.sock_to_prev_prop = self.sock_to_prev.accept()
        logger_wp.info("Created socket to previous node")

        if VERB:
            print("-> Done!                     ")
        self.running = True

        if self.node_type != "starter":
            # Open server towards next node
            if VERB:
                print(
                    f"[INFO] Opening socket to next node (to port {self.next_node['inference']['port_in']})"
                )

            self._start_client()
            assert self.sock_to_next is not None
            logger_wp.info("Created socket to next node")

            if VERB:
                print("-> Done!                     ")

    def shutdown(self) -> int:
        """
        Turn off the node - stop server, close sockets and stop thread.

        Returns:
            1 upon success, 0 otherwise
        """
        try:
            time.sleep(2)
            cp.engine.stop()
            self.sock_to_prev_prop[0].close()
            self.sock_to_prev.close()
            self.sock_to_next.close()
            self.running = False  # Redundant
            self.queue_thread.join()
            if self.node_type != "starter":
                self._running_thread.join()
            return 1
        except:
            return 0

    # ----- Private -----------------------------------------------------------

    def _load_tokenizer(
        self,
    ) -> Union[CharacterTokenizer, BPETokenizer, tiktoken.Encoding]:
        """
        Load the tokenizer information from the path specified in class attribute
        `self.tok_meta_path`.
        The tokenizer object will be stored in `self.tok`.

        Returns:
            the tokenizer object
        """
        if self.tok_meta_path is not None:
            logger_wp.info(f"Loading tokenizer metadata from {self.tok_meta_path}")
            if VERB:
                print(f"[INFO] Loading tokenizer metadata from {self.tok_meta_path}")
        else:
            logger_wp.info("Loading GPT-2 tokenizer (50k)")
            if VERB:
                print("[INFO]: loading GPT-2 tokenizer")

        if self.tok_meta_path.endswith(".pkl"):
            with open(self.tok_meta_path, "rb") as f:
                meta = pickle.load(f)
            self.tok = CharacterTokenizer(meta["stoi"], meta["itos"])
        elif os.path.isdir(self.tok_meta_path):
            vocab_path = os.path.join(self.tok_meta_path, "encoder.json")
            merges_path = os.path.join(self.tok_meta_path, "merges.bpe")
            self.tok = BPETokenizer(vocab_path, merges_path)
        else:
            self.tok = tiktoken.get_encoding("gpt2")  # Class: tiktoken.Encoding

        return self.tok

    def _start_server(self, max_tries: int = 30):
        """
        Start the server socket, i.e., the socket to the previous node in the chain.
        """
        loopsigns = ["|", "/", "-", "\\"]
        self.sock_to_prev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        failed = True
        tries = 0
        while failed and tries < max_tries:
            # Attempt to bind
            try:
                self.sock_to_prev.bind(
                    (
                        self.node_config["addr"],
                        self.node_config["inference"]["port_in"],
                    )
                )
            except:
                tries += 1
                if VERB:
                    print(f"[INFO] Retrying {loopsigns[tries % 4]}", end="\r")
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
        Start the client socket, i.e., the socket to the next node in the chain.
        """
        loopsigns = ["|", "/", "-", "\\"]
        conn = False
        tries = 0
        while not conn and tries < max_tries:
            try:
                self.sock_to_next = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Bind should work even after some fails
                self.sock_to_next.bind(
                    (
                        self.node_config["addr"],
                        self.node_config["inference"]["port_out"],
                    )
                )
                self.sock_to_next.connect(
                    (
                        self.next_node["addr"],
                        self.next_node["inference"]["port_in"],
                    )
                )
                if VERB:
                    print("Connected to next node!")
            except:
                # Can either fail when binding or when connecting
                tries += 1
                if VERB:
                    print(f"[INFO] Retrying {loopsigns[tries % 4]}", end="\r")
                time.sleep(1)
            else:
                conn = True

        if not conn:
            raise ConnectionError(
                f"Unable to create client socket at ({self.node_config['addr']}, {self.node_config['inference']['port_in']})"
            )

    def _fill_queue(self):
        """
        This method has the goal of managing incoming messages from previous nodes in
        the chain.
        As a message is received, its contents are stored in the message queue
        (`self.message_queue`).
        This allows to store locally each of the received messages, in order.
        The order is crucial for the correct functioning of the program (pipelining).

        This method loops infinitely and constantly waits for incoming messages.
        For this reason, it is ran on a separate thread, and it is stopped when the main
        thread, running the processing function, finishes.
        """
        assert self.sock_to_prev is not None and self.sock_to_prev_prop != ()

        _n_recv_msg = 0
        while self.running:
            # Receive information from the new socket (exact length)
            msg = self.recv_from_prev(HEADERLENGTH)

            # Extract message length from the header
            msg_len = int(msg[:HEADERLENGTH])
            _n_recv_msg += 1

            # Read payload (exact size - this is important)
            msg_payload = self.recv_from_prev(msg_len)
            data = pickle.loads(msg_payload)
            logger_wp.debug(f"Received full message {_n_recv_msg} of length {msg_len}")

            # Look for stopping msg
            if "stop" in data and data["stop"]:
                # Stopping sequence
                if VERB:
                    print("Stopping message received! Generation complete!")
                logger_wp.info("Stopping message received! Generation complete!")
                self.running = False
            else:  # Not here if stopping message is received
                self.message_queue.append(data)

    def _starter_loop(self, max_new_tokens: int) -> Tuple[List[str], float]:
        """
        Generation loop for the starter node only.
        This loop has a finite duration, as the starter knows what is the length of the
        samples to be generated.

        Args:
            max_new_tokens: maximum number of tokens

        Returns:
            list containing the `n_nodes` generated samples
            total generation time in seconds
        """
        # TODO: allow to generate as many samples as desired (>= n. nodes) - see #10
        assert self.model_config is not None and self.model is not None

        if "cuda" in self.device:
            device_type = "cuda"
        elif "mps" in self.device:
            device_type = "mps"
        else:
            device_type = "cpu"
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[DTYPE]
        ctx = (  # Use autocast if on cuda or cpu (MPS not supported yet)
            nullcontext()
            if device_type == "mps"
            else torch.autocast(device_type=device_type, dtype=ptdtype)
        )

        # Encode starting sequence (TODO: implement prompt support - different
        # prompts for different samples - see #12)
        start = "\n"
        start_ids = self.tok_encode(start)
        idx = [
            torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
            for _ in range(self.n_nodes)
        ]

        self.model.eval()
        start_time = time.time()
        count_wait = 0  # Count the number of times the loop had to wait
        if PLOTS:
            self.tok_time.append((0, 0))
        with torch.no_grad():
            with ctx:
                total_iters = max_new_tokens * self.n_nodes
                for k in range(total_iters):
                    logger_wp.info(f"Iter {k}")
                    print(
                        f"Generating: {loading_bar(k, total_iters, 20)} ({k}/{total_iters})",
                        end="\r",
                    )
                    if PLOTS:
                        self.tok_time.append((k, time.time() - start_time))
                    sample_id = k % self.n_nodes  # Which of the n_nodes samples

                    if k >= self.n_nodes:
                        # We are not in the first iteration (k starts from 0)
                        # can start processing messages from last secondary node
                        old_count_w = count_wait
                        while len(self.message_queue) <= 0:
                            count_wait += 1
                        if count_wait - old_count_w > 0:
                            logger_wp.warn(
                                f"Iter {k} - Had to wait for queue to fill up!"
                            )
                        in_msg = self.message_queue.popleft()
                        sample_in = in_msg["sample_index"]

                        # Check correct order
                        assert (
                            sample_in == sample_id
                        ), f"> ITER [{k}] - Received sample ID: {sample_in}, expected ID: {sample_id}"

                        idx_from_fin = in_msg["data"].to(self.device)
                        logits = self.model.forward_last(idx_from_fin)
                        logits = logits[:, -1, :] / self.temperature
                        if self.top_k is not None:
                            v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float("Inf")
                        probs = F.softmax(logits, dim=1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx[sample_id] = torch.cat((idx[sample_id], idx_next), dim=1)

                    # Send to next iff not at the last token
                    if k < (self.n_nodes * (max_new_tokens - 1)):
                        # Crop to block size
                        idx_cond = (
                            idx[sample_id]
                            if idx[sample_id].size(1) <= self.model_config.block_size
                            else idx[sample_id][:, -self.model_config.block_size :]
                        )
                        # Forward in local model
                        idx_cond = self.model(idx_cond)

                        # Build message
                        out_msg = self._build_msg(idx_cond, sample_id)
                        self.send_to_next(out_msg)

        tot_time = time.time() - start_time
        if PLOTS:
            self.tok_time.append((total_iters, tot_time))
            # Store plotted points as csv file
            points_file_path = os.path.join(
                script_dir,
                "..",
                "logs",
                "tok-per-time",
                f"tokens_time_samples_mdi_{MODEL_TYPE}_{self.n_nodes}samples.csv",
            )
            if not os.path.exists(os.path.dirname(points_file_path)):
                os.mkdir(os.path.dirname(points_file_path))
            with open(points_file_path, "w") as f:
                times = [x[1] for x in self.tok_time]
                n_samples = [x[0] for x in self.tok_time]
                for i in range(len(times)):
                    f.write(f"{times[i]},{n_samples[i]}\n")

            plot_tokens_per_time(
                self.tok_time,
                out_path=os.path.join(
                    script_dir, "..", "img", f"tokens_time_mdi_{MODEL_TYPE}.png"
                ),
            )

        # Send stop message to the next
        self.send_to_next(self.stop_msg)
        logger_wp.info("Generation completed")
        if VERB:
            print("[INFO] Generation completed!                          ")
            print(f"> Total time for generation: {tot_time} s")
            print(
                f"Total time spent waiting: {count_wait}*0.01 = {count_wait * 0.01} s"
            )

        return [self.tok_decode(smp[0].tolist()) for smp in idx], tot_time

    def _node_loop(self):
        """
        Execution loop for non-starter nodes. This method must be used as the target of
        a thread that is launched once the node has been correctly initialized.

        The execution will be stopped once a PUT request is made to /stop.
        """
        assert self.sock_to_prev is not None and self.sock_to_next is not None
        assert self.model is not None and self.model_config is not None

        # Should be overrided by kwargs
        if "cuda" in self.device:
            device_type = "cuda"
        elif "mps" in self.device:
            device_type = "mps"
        else:
            device_type = "cpu"
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[DTYPE]
        ctx = (  # Use autocast if on cuda or cpu (MPS not supported yet)
            nullcontext()
            if device_type == "mps"
            else torch.autocast(device_type=device_type, dtype=ptdtype)
        )

        self.model.eval()
        loopsigns = ["|", "/", "-", "\\"]
        iter = 0
        exp_ind = 0  # Expected sample index from previous
        count_wait = 0  # Count the number of times the loop had to wait
        with torch.no_grad():
            with ctx:
                while self.running:
                    logger_wp.info(f"Iter {iter}")
                    old_count_w = count_wait
                    while len(self.message_queue) <= 0:  # Wait for messages
                        count_wait += 1
                    if count_wait - old_count_w > 0:
                        logger_wp.warn(
                            f"Iter {iter} - Had to wait for queue to fill up!"
                        )
                    # Extract message from queue
                    in_msg = self.message_queue.popleft()
                    # Unpack
                    samp_ind = in_msg["sample_index"]
                    assert (
                        exp_ind == samp_ind
                    ), f"Expected sample index {exp_ind}, received {samp_ind}"
                    exp_ind = (samp_ind + 1) % self.n_nodes

                    ins = in_msg["data"].to(self.device)
                    if self.running:
                        print(f"> Generating {loopsigns[iter % 4]}", end="\r")
                        # Forward pass
                        outs = self.model(ins)
                        # Build msg
                        out_msg = self._build_msg(outs, samp_ind)
                        # Send to next
                        self.send_to_next(out_msg)
                        iter += 1
                    else:
                        print("> Generation completed!")
                        print(f"Total times waited: {count_wait}")
                        self.send_to_next(self.stop_msg)

    def _build_msg(self, data, sample_index) -> Dict:
        """
        Build the message which is transmitted to the next node.

        Args:
            data: the activations to be transmitted
            sample_index: index of the current sample (allows to check)

        Returns:
            the message - a Python dict with the fields "sample_index" and
            "data"
        """
        return {"sample_index": sample_index, "data": data}

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
            Receive configuration info from the starter node and start connection with
            previous and next, then start generation, i.e., wait for incoming data
            through the sockets to be passed through the local model chunk.
        """
        if self.node_type is None and self.model is None:  # Only for non-init nodes
            if len(path) > 0 and path[0] == "init":
                assert not self.running
                init_msg = pickle.loads(cp.request.body.read())
                self.node_type = init_msg["role"]
                self.prev_node = init_msg["prev_node"]
                self.next_node = init_msg["next_node"]
                self.model_config = GPTConfig(**init_msg["model_config"])
                if "params" in init_msg:
                    if VERB:
                        print("Received parameters from starter")
                    self.model_params = init_msg["params"]
                else:
                    assert self.chunk_path is not None
                    if VERB:
                        print("Loading parameters from disk")
                    chunk_cont = torch.load(self.chunk_path, map_location=self.device)
                    # Check compatibility (all keys of chunk_cont should be in init_msg)
                    assert all(
                        [
                            k in init_msg["model_config"]
                            for k in chunk_cont["model_args"]
                        ]
                    ), f"Different settings:\n{chunk_cont['model_args']}\n\n{init_msg['model_config']}"
                    self.model_params = chunk_cont["model"]
                self.n_nodes = init_msg["n_nodes"]
                # Set up the node
                self.init_model(init_msg["n_layers"])
                if VERB:
                    print(f"[INFO] Starting operation - {self.node_type} node")
                logger_wp.info("Received initialization information!")
                self._running_thread = threading.Thread(target=self.start, daemon=True)
                self._running_thread.start()
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found")
        elif self.model is None:
            raise cp.HTTPError(
                403,
                f"Failed to configure node - the model was already initialized: {self.node_type}",
            )

    def PUT(self, *path):
        """
        Used by the starter to stop running nodes at the end of the generation.
        """
        if self.node_type != "secondary":
            raise cp.HTTPError(501, "PUT not implemented!")
        else:
            if len(path) > 0 and path[0] == "stop":
                self._end_thr = threading.Thread(target=self.shutdown, daemon=True)
                self._end_thr.start()
                if VERB:
                    print("[INFO] Node stopped!")
                logger_wp.info("Received stopping directive")
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found!")

    def DELETE(self):
        """Not implemented"""
        raise cp.HTTPError(501, "DELETE not implemented!")


class GPTDistributed:
    """
    Distributed implementation of a minimal GPT2 instance.
    """

    # Syntax of the message used to initialize other nodes
    init_msg = {
        "role": "",  # Role name
        # "params": {},  # State dict  # -- REMOVED to allow loading chunks
        "model_config": GPTConfig(),
        "n_nodes": 0,
        "prev_node": {},  # From .json
        "next_node": {},  # From .json
        "n_layers": 0,  # Number of transformer layers
        "device": "cpu",
    }

    def __init__(
        self,
        ckpt_path: str,
        nodes_info_path: Union[str, None] = None,
        model_was_split: bool = False,
        **kwargs,
    ):
        """
        Instantiate a GPTDistributed object to perform model-distributed inference.

        Args:
            ckpt_path: path of the full model (pretrained) or GPT2 flavor from HF
            nodes_info_path: path of the configuration JSON - if not provided, a default
                one is used
            model_was_split: true if the model chunks have already been generated (the
                starter node does not need to split the model himself)
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

        self.mod_split = model_was_split

        # Override global constants
        if "verb" in kwargs:
            global VERB
            print("Overriding 'verb'")
            VERB = bool(kwargs["verb"])
        if "plots" in kwargs:
            global PLOTS
            print("Overriding 'plots'")
            PLOTS = bool(kwargs["plots"])
        if "device" in kwargs:
            global DEVICE
            print("Overriding 'device'")
            DEVICE = str(kwargs["device"])

        with open(settings_path, "r") as f:
            self.nodes_info = json.load(f)
            f.close()

        logger_wp.info("Loaded nodes information JSON file!")

        # Store own information:
        self.own_config = self.nodes_info["nodes"]["starter"]
        self.own_addr = self.own_config["addr"]
        self.own_comm_port = self.own_config["communication"]["port"]
        self.own_inference_port_in = self.own_config["inference"]["port_in"]
        self.own_inference_port_out = self.own_config["inference"]["port_out"]

        # Get the model parameters and split them based on n. of nodes
        self.n_secondary = len(self.nodes_info["nodes"]["secondary"])
        if self.n_secondary < 1:
            raise ValueError("No secondary nodes provided!")
        self.n_total_nodes = 1 + self.n_secondary

        if os.path.exists(ckpt_path):
            # Either load full model, or load chunk
            try:
                self.model_ckpt = torch.load(ckpt_path, map_location=DEVICE)
            except:
                # Erase variable to effectively free memory
                try:
                    del self.model_ckpt
                except AttributeError:
                    pass  # model_ckpt not defined (good)
                gc.collect()
                # It may be that the model does not fit all in the VRAM
                if VERB:
                    print("Loading full model on RAM - not enough VRAM")
                logger_wp.warn("Loading full model on RAM - not enough VRAM")
                self.model_ckpt = torch.load(ckpt_path, map_location="cpu")
        elif ckpt_path in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}:
            # HF pretrained
            assert not model_was_split
            model_sd, model_args = load_from_hf(ckpt_path)
            self.model_ckpt = {"model": model_sd, "model_args": model_args}
        else:
            raise ValueError(f"Unrecognized model: {ckpt_path}")

        if not model_was_split:
            if VERB:
                print("Loading full model and splitting it into chunks")
            # Full model loaded: extract state dict (complete model)
            self.complete_model = self.model_ckpt["model"]  # State dict
            self.starter_has_entire_mod = True

            # Remove problematic keys
            # NOTE: this shouldn't happen anymore (it was a problem in nanoGPT)
            unwanted_prefix = "_orig_mod."
            for k in list(self.complete_model):
                if k.startswith(unwanted_prefix):
                    self.complete_model[
                        k[len(unwanted_prefix) :]
                    ] = self.complete_model.pop(k)

            # Split model - NOTE: the function removes the elements from
            # self.complete_model, saving memory (no duplicate values)
            self.model_chunks, self.layers_info = split_parameters(
                model_params=self.complete_model, n_nodes=self.n_total_nodes
            )
            # FIXME: this assert may fail since we removed attn.bias from new ckpts
            # Or it may not... attn.bias is passed over and not used
            assert (
                len(self.complete_model) == 0
            ), "Something went wrong when splitting model - leftover parameters!"
            del self.complete_model
            gc.collect()
        else:
            if VERB:
                print("Model was split in advance - loading starter chunk")
            # CKPT chunk components:
            # - model: actual model chunk (params)
            # - model_args: model parameters (to be passed to GPTConfig after)
            # - config: globals of training - maybe not needed...
            # - dist_config: layer_info - n. of layers for each node
            self.starter_has_entire_mod = False
            self.model_chunks = {"starter": self.model_ckpt["model"]}
            self.layers_info = self.model_ckpt["dist_config"]

        self.n_layers_tot = (
            self.layers_info["N_LAYERS_START"]
            + self.n_secondary * self.layers_info["N_LAYERS_SECONDARY"]
        )

        global MODEL_TYPE
        embd = self.model_ckpt["model_args"]["n_embd"]
        ctx = self.model_ckpt["model_args"]["block_size"]
        MODEL_TYPE = f"{self.n_layers_tot}layers_{ctx}ctx_{embd}embd"

        # Extract tokenizer metadata information (if any) and locate it
        dataset_dir = None
        dataset_name = None
        if "config" in self.model_ckpt and "DATASET_PATH" in self.model_ckpt["config"]:
            dataset_dir = os.path.normpath(self.model_ckpt["config"]["DATASET_PATH"])
            dataset_name = os.path.basename(dataset_dir)
        elif "config" in self.model_ckpt and "DATASET" in self.model_ckpt["config"]:
            dataset_name = os.path.basename(
                os.path.normpath(self.model_ckpt["config"]["DATASET"])
            )
            dataset_dir = os.path.join(script_dir, "..", "data", dataset_name)

        if dataset_name is not None and dataset_dir is not None:
            if os.path.exists(os.path.join(dataset_dir, "meta.pkl")):
                self.tok_meta_path = os.path.join(dataset_dir, "meta.pkl")
                if VERB:
                    print(f"Using character-level tokenizer ({self.tok_meta_path})")
            elif os.path.exists(
                os.path.join(dataset_dir, "encoder.json")
            ) and os.path.exists(os.path.join(dataset_dir, "merges.bpe")):
                self.tok_meta_path = dataset_dir
                if VERB:
                    print(f"Using BPE tokenizer found in {dataset_dir}")
            else:
                self.tok_meta_path = None
                if VERB:
                    print("No tokenizer information found, assuming GPT-2 encodings...")
        else:
            self.tok_meta_path = None
            if VERB:
                print("No tokenizer information found, assuming GPT-2 encodings...")

        self.model_config = GPTConfig(**self.model_ckpt["model_args"])

        # Create webserver
        starter_config = self.init_msg.copy()
        starter_config["role"] = "starter"
        starter_config["params"] = self.model_chunks["starter"]
        starter_config["model_config"] = self.model_config
        starter_config["n_nodes"] = self.n_total_nodes
        starter_config["n_layers"] = self.layers_info["N_LAYERS_START"]
        starter_config["prev_node"] = self.nodes_info["nodes"]["secondary"][-1]
        starter_config["next_node"] = self.nodes_info["nodes"]["secondary"][0]
        starter_config["tok_metadata_path"] = self.tok_meta_path
        # Device selection: use default if not found in configuration JSON file
        starter_config["device"] = (
            DEVICE if "device" not in self.own_config else self.own_config["device"]
        )

        self.webserv = GPTServer(
            node_type="starter",
            node_config=self.own_config,
            starter_config=starter_config,
        )

    def configure_nodes(self) -> int:
        """
        Send POST requests to the other nodes to inform them of their role and including
        their chunk of model.

        Information sent:
            - Node role ("role")
            - Model config (GPTConfig as dict) ("model_config")
            - Model parameters ("params") - from pickle.dumps() - if not split before
            - Previous node information - from json file ("prev_node")
            - Next node information - from json ("next_node")

        Returns:
            1 if success
            0 if at least 1 node fails
        """
        out = 1  # Return code

        # Store the prev and next in a smart way
        prev = self.nodes_info["nodes"]["starter"]
        if self.n_secondary == 1:
            next = self.nodes_info["nodes"]["starter"]
        elif self.n_secondary > 1:
            next = self.nodes_info["nodes"]["secondary"][1]
        else:
            raise ValueError("Should not be here!")

        # Secondary nodes config
        for i, sec_node in enumerate(self.nodes_info["nodes"]["secondary"]):
            if VERB:
                print(f"Initializing secondary node n.{i}")

            curr_msg = self.init_msg.copy()
            curr_msg["role"] = "secondary"
            curr_msg["model_config"] = self.model_config.asdict()
            if not self.mod_split:
                curr_msg["params"] = self.model_chunks["secondary"][i]
            curr_msg["n_nodes"] = self.n_total_nodes

            curr_msg["prev_node"] = prev
            curr_msg["next_node"] = next

            curr_msg["n_layers"] = self.layers_info["N_LAYERS_SECONDARY"]
            curr_msg["device"] = (
                DEVICE if "device" not in sec_node else sec_node["device"]
            )

            # Update next and prev for next iteration
            prev = sec_node
            if i == self.n_secondary - 1:  # Last iter in loop - finished
                next = None
            elif i == self.n_secondary - 2:  # Second to last iter
                next = self.nodes_info["nodes"]["starter"]
            else:
                next = self.nodes_info["nodes"]["secondary"][i + 2]

            # Send POST request
            target_addr = sec_node["addr"]
            target_port = sec_node["communication"]["port"]

            addr = f"http://{target_addr}:{target_port}/init"
            out *= self.request_to_node("post", addr, curr_msg)

            if not out:
                if VERB:
                    print("> Failed!")
                logger_wp.error(f"Failed to initialize secondary node {i}!")
                return out

            if VERB:
                print("> Success!")
            logger_wp.info(f"Secondary node {i} was initialized successfully")

        return out

    def request_to_node(
        self, req_type: str, addr: str, content: Any, max_n_requests: int = 100
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
        if VERB:
            print(f"Sending {req_type} request to {addr}")
            print(f"Payload: {len(pickle.dumps(content))} Bytes")
        try:
            # Specify timeout
            ret = req_func(
                addr,
                data=pickle.dumps(content),
                timeout=100,
            )

            if ret.status_code == 413:
                raise ConnectionError(f"Max payload for {req_type} was exceeded!")
            logger_wp.debug(
                f"Successful {req_type} request sent to {addr} - code {ret.status_code}"
            )
        except requests.exceptions.Timeout:
            if VERB:
                print("Connection timed out!")
            logger_wp.warning(f"Request timed out!")
            n_ret += 1
        except:
            logger_wp.warning(f"Unable to submit {req_type} request sent to {addr}")
            n_ret += 1
        while (ret is None or ret.status_code != 200) and n_ret < max_n_requests:
            if VERB:
                print(
                    f"Unable to reach node ({addr}) - retrying in 2s ({n_ret}/{max_n_requests})"
                )
            time.sleep(2)
            try:
                ret = req_func(
                    addr,
                    data=pickle.dumps(content),
                    timeout=10000,
                )
                logger_wp.debug(
                    f"Successful {req_type} request sent to {addr} - code {ret.status_code}"
                )
            except requests.exceptions.Timeout:
                if VERB:
                    print("Connection timed out!")
                logger_wp.warning(f"Request timed out!")
            except:
                logger_wp.warning(f"Unable to submit {req_type} request sent to {addr}")
            n_ret += 1

        if ret is not None and ret.status_code == 200:
            return 1
        return 0

    def start(self, tokens_per_sample: int = 1000) -> Tuple[List[str], float]:
        """
        Start the operation - webserver + model

        Stop when the model finished running, i.e., all tokens have been
        generated.

        This method calls back self.configure_nodes() and self.webserv.start()

        Args:
            tokens_per_sample: number of generated tokens per sample; the number
                of samples is the same as the number of nodes

        Returns:
            List of produced samples
            Total generation time in seconds
        """
        # TODO: add assertions (uninitialized values)
        if not self.configure_nodes():
            raise AssertionError("Unable to initialize required nodes!")

        # The code below assumes we will receive the correct info (not None)
        out, time_gen = self.webserv.start(max_new_tokens=tokens_per_sample)

        print("-------------------------------------------------")
        print("Produced output:\n")
        for i, smpl in enumerate(out):
            print("-------------------------------------------------")
            print(f"Sample {i + 1}:")
            print(smpl, "\n")
        print("-------------------------------------------------")

        # Once finished, send PUT to each node to terminate execution for them
        self.stop()

        return out, time_gen

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
        for sec_node in self.nodes_info["nodes"]["secondary"]:
            target_addr = sec_node["addr"]
            target_port = sec_node["communication"]["port"]

            addr = f"http://{target_addr}:{target_port}/stop"
            if VERB:
                print(f"Sending PUT request to {addr}")
            out *= self.request_to_node("PUT", addr, {})

        self.webserv.shutdown()

        return out
