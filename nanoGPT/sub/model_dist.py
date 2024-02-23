#!/usr/bin/env python3

import json
import logging
import os
import pickle
import socket
import threading
import time
import warnings
from typing import Any, Dict, List, Mapping, Tuple, Union

import cherrypy as cp
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.char_tokenizer import CharacterTokenizer
from sub.config import DEVICE, HEADERLENGTH, PLOTS, TEMPERATURE, TOP_K, VERB
from sub.model import Block, GPTConfig, LayerNorm
from sub.server_config import MAX_TRIES
from sub.utils import (deserialize_params, loading_bar, plot_tokens_per_time,
                       serialize_params, split_parameters)

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
# Logging
script_dir = os.path.dirname(__file__)
logger_wp = logging.getLogger("model_dist")
logger_wp.setLevel(logging.NOTSET)

MODEL_TYPE = ""


class StarterNode(nn.Module):
    """Starter node"""

    params_init = False

    def __init__(self, config: GPTConfig, n_transf_layers: int):
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
                layers=nn.ModuleList(
                    [Block(config) for _ in range(n_transf_layers)]
                ),
            )
        )

    def load_weights(self, params: Mapping[str, Any]) -> int:
        """Load weights"""
        self.load_state_dict(params)
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
        tok_emb = self.starter_model.token_embedding(idx)

        # Obtain positional embeddings by encoding values (0, ..., t)
        pos_emb = self.starter_model.position_embedding(
            torch.arange(t, device=device)
        )

        x = self.starter_model.drop(tok_emb + pos_emb)  # (B, T, C)

        for block in self.starter_model.layers:
            x = block(x)

        return x


class IntermediateNode(nn.Module):
    """Intermediate node"""

    params_init = False

    def __init__(self, config: GPTConfig, n_transf_layers: int):
        super().__init__()
        assert config.vocab_size is not None

        self.config = config

        # Follow naming convention
        self.intermediate_model = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [Block(config) for _ in range(n_transf_layers)]
                )
            )
        )

    def load_weights(self, params: Mapping[str, Any]) -> int:
        """Load weights"""
        self.load_state_dict(params)
        self.params_init = True
        if VERB:
            print(f"Weights loaded!")
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

    def __init__(self, config: GPTConfig, n_transf_layers: int):
        super().__init__()
        assert config.vocab_size is not None

        self.config = config

        self.finisher_model = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [Block(config) for _ in range(n_transf_layers)]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
                lm_head=nn.Linear(config.n_embd, config.vocab_size),
            )
        )

    def load_weights(self, params: Mapping[str, Any]) -> int:
        """Load weights"""
        self.load_state_dict(params)
        self.params_init = True
        if VERB:
            print(f"Weights loaded!")
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
    stop_msg = {"stop": True}
    sock_to_prev: Union[socket.socket, None] = None
    sock_to_prev_prop: Tuple = ()  # NOTE: used now
    sock_to_next: Union[socket.socket, None] = None
    sock_to_next_prop: Tuple = ()  # NOTE: not used!

    # Message queue
    message_queue = []

    # Some model configs:
    top_k = TOP_K
    temperature = TEMPERATURE

    # Stats - n. tokens/time (tuples)
    tok_time = []

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

            self.init_model(starter_config["n_layers"], set_eval=True)

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
                "server.thread_pool": 8,
                # remove any limit on the request body size; default is 100MB
                "server.max_request_body_size": 0,
                # increase server socket timeout to 60s; default is 10s
                "server.socket_timeout": 60,
            }
        )

        cp.engine.start()

    # ----- Public ------------------------------------------------------------

    def init_model(self, n_transf_layers: int, set_eval: bool = True):
        """
        Initialize the node's model chunk, passing the parameters.

        The model will also be moved to the target device.

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
            self.model = StarterNode(self.model_config, n_transf_layers)
        elif self.node_type == "intermediate":
            self.model = IntermediateNode(self.model_config, n_transf_layers)
        elif self.node_type == "finisher":
            self.model = FinisherNode(self.model_config, n_transf_layers)
        else:
            raise ValueError(f"Unsupported node type {self.node_type}")

        self.model = self.model.to(self.model_config.device)
        self.model.load_weights(self.model_params)
        if set_eval:
            # Set to evaluation mode (no backpropagation)
            self.model.eval()

    def start(
        self,
        max_new_tokens: Union[None, int] = None,
    ) -> Union[None, Tuple[List[str], float]]:
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

        Returns:
            if starter node, return the list of produced samples, else nothing
        """
        assert self.sock_to_prev is None and self.sock_to_next is None
        assert self.next_node is not None and self.prev_node is not None
        assert self.model_config is not None and self.model is not None

        # Configuration for all nodes
        self.create_sockets()

        assert self.sock_to_prev is not None and self.sock_to_next is not None

        self.message_queue = []  # Initialize empty queue

        # Differentiate between different types
        if self.node_type == "starter":
            assert max_new_tokens is not None

            self._load_tokenizer()
            self.tok_encode = lambda s: self.tok.encode(s)
            self.tok_decode = lambda l: self.tok.decode(l)

            if VERB:
                print("[INFO] Tokenizer loaded!")
                print("[INFO] Starting queue thread")
            logger_wp.info("Tokenizer loaded!")
            logger_wp.info("Starting queue thread")

            self.queue_thread = threading.Thread(
                target=self._fill_queue, daemon=True
            )
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
            self.queue_thread = threading.Thread(
                target=self._fill_queue, daemon=True
            )
            self.queue_thread.start()

            if VERB:
                print("[INFO] Starting generation loop")
            logger_wp.info("Starting generation loop")
            self._node_loop()

    def recv_from_prev(self, size: int) -> bytes:
        """
        Receive a message of the specified size from the previous node.

        Remark: the size specified in socket.recv(<>) is the MAX size that will
        be read from the receiver buffer.

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

        The message is composed by a header of HEADERLENGTH bytes including the
        length of the actual message, plus a message of MSGLENGTH bytes
        containing the zero-padded message.
        """
        assert self.sock_to_next is not None

        message_str = pickle.dumps(data)
        tx_msg = (
            bytes(f"{len(message_str):<{HEADERLENGTH}}", "utf-8") + message_str
        )
        # NOTE: attempt at sending multiple messages in a "safe" way (no sendall)
        while tx_msg:
            tx_msg = tx_msg[self.sock_to_next.send(tx_msg) :]
        logger_wp.debug("Sent full message to next")
        # self.sock_to_next.sendall(tx_msg)

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
        self.sock_to_prev.listen(1)

        self.sock_to_prev_prop = self.sock_to_prev.accept()
        # self.sock_to_prev_prop[0].setsockopt(
        #     socket.SOL_SOCKET, socket.SO_RCVBUF, MSGLENGTH
        # )
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

    def _load_tokenizer(self) -> CharacterTokenizer:
        """
        Load the tokenizer information from the path specified in class
        attribute `self.tok_meta_path`.

        The tokenizer object will be stored in `self.tok`.

        Returns:
            the tokenizer object
        """
        if VERB:
            print(
                f"[INFO] Loading tokenizer metadata from {self.tok_meta_path}"
            )
        logger_wp.info(f"Loading tokenizer metadata from {self.tok_meta_path}")
        with open(self.tok_meta_path, "rb") as f:
            meta = pickle.load(f)
        self.tok = CharacterTokenizer(meta["stoi"], meta["itos"])

        return self.tok

    def _start_server(self, max_tries: int = 30):
        """
        Start the server socket, i.e., the socket to the previous node in the
        chain.
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
                self.sock_to_next = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM
                )
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
        This method has the goal of managing incoming messages from previous
        nodes in the chain.
        As a message is received, its contents are stored in the message queue
        (`self.message_queue`).
        This allows to store locally each of the received messages, in order.
        The order is crucial for the correct functioning of the program
        (pipelining).

        This method loops infinitely and constantly waits for incoming messages.
        For this reason, it is ran on a separate thread, and it is stopped when
        the main thread, running the processing function, finishes.
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
            logger_wp.debug(
                f"Received full message {_n_recv_msg} of length {msg_len}"
            )

            # Look for stopping msg
            if "stop" in data and data["stop"]:
                # Stopping sequence
                if VERB:
                    print("Stopping message received! Generation complete!")
                logger_wp.info(
                    "Stopping message received! Generation complete!"
                )
                self.running = False
            else:  # Not here if stopping message is received
                self.message_queue.append(data)

    def _starter_loop(self, max_new_tokens: int) -> Tuple[List[str], float]:
        """
        Generation loop for the starter node only.
        This loop has a finite duration, as the starter knows what is the length
        of the samples to be generated.

        Args:
            max_new_tokens: maximum number of tokens

        Returns:
            list containing the `n_nodes` generated samples
            total generation time in seconds
        """
        assert self.model_config is not None and self.model is not None

        # Encode starting sequence (TODO: implement prompt support - different
        # prompts for different samples)

        start = "\n"
        start_ids = self.tok_encode(start)
        idx = [
            torch.tensor(
                start_ids, dtype=torch.long, device=self.model_config.device
            )[None, ...]
            for _ in range(self.n_nodes)
        ]

        start_time = time.time()
        count_wait = 0  # Count the number of times the loop had to wait
        if PLOTS:
            self.tok_time.append((0, 0))
        with torch.no_grad():
            # with CTX:  # FIXME
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
                    # can start processing messages from finisher
                    old_count_w = count_wait
                    while len(self.message_queue) <= 0:
                        count_wait += 1
                        # time.sleep(0.01)
                    if count_wait - old_count_w > 0:
                        logger_wp.warn(
                            f"Iter {k} - Had to wait for queue to fill up!"
                        )
                    in_msg = self.message_queue.pop(0)
                    sample_in = in_msg["sample_index"]

                    # Check correct order
                    assert (
                        sample_in == sample_id
                    ), f"> ITER [{k}] - Received sample ID: {sample_in}, expected ID: {sample_id}"

                    out_logits = in_msg["data"].to(self.model_config.device)
                    logits = out_logits[:, -1, :] / self.temperature
                    if self.top_k is not None:
                        v, _ = torch.topk(
                            logits, min(self.top_k, logits.size(-1))
                        )
                        logits[logits < v[:, [-1]]] = -float("Inf")
                    probs = F.softmax(logits, dim=1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx[sample_id] = torch.cat(
                        (idx[sample_id], idx_next), dim=1
                    )

                if k < (self.n_nodes * (max_new_tokens - 1)):
                    # Send to next iff not at the last token
                    # Crop to block size
                    idx_cond = (
                        idx[sample_id]
                        if idx[sample_id].size(1)
                        <= self.model_config.block_size
                        else idx[sample_id][:, -self.model_config.block_size :]
                    )
                    # Forward in local model
                    idx_cond = self.model(idx_cond)

                    # Build message
                    out_msg = self._build_msg(idx_cond, sample_id)
                    self.send_to_next(out_msg)
                    # Sleep for 1 ms - do not overwhelm receiver
                    # time.sleep(0.001)

        tot_time = time.time() - start_time
        if PLOTS:
            self.tok_time.append((total_iters, tot_time))
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
        Execution loop for non-starter nodes. This method must be used as the
        target of a thread that is launched once the node has been correctly
        initialized.

        The execution will be stopped once a PUT request is made to /stop.
        """
        assert self.sock_to_prev is not None and self.sock_to_next is not None
        assert self.model is not None and self.model_config is not None

        loopsigns = ["|", "/", "-", "\\"]
        iter = 0
        exp_ind = 0  # Expected sample index from previous
        count_wait = 0  # Count the number of times the loop had to wait
        with torch.no_grad():
            while self.running:
                logger_wp.info(f"Iter {iter}")
                old_count_w = count_wait
                while len(self.message_queue) <= 0:  # Wait for messages
                    count_wait += 1
                    # time.sleep(0.01)
                if count_wait - old_count_w > 0:
                    logger_wp.warn(
                        f"Iter {iter} - Had to wait for queue to fill up!"
                    )
                # Extract message from queue
                in_msg = self.message_queue.pop(0)
                # Unpack
                samp_ind = in_msg["sample_index"]
                assert (
                    exp_ind == samp_ind
                ), f"Expected sample index {exp_ind}, received {samp_ind}"
                exp_ind = (samp_ind + 1) % self.n_nodes

                ins = in_msg["data"].to(self.model_config.device)
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
                    print(
                        f"Total time spent waiting: {count_wait}*0.01 = {count_wait * 0.01} s"
                    )
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
                self.n_nodes = init_msg["n_nodes"]
                # Set up the node
                self.init_model(init_msg["n_layers"])
                if VERB:
                    print(f"[INFO] Starting operation - {self.node_type} node")
                logger_wp.info("Received initialization information!")
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
    Distributed implementation of a minimal GPT2 instance
    """

    # Syntax of the message used to initialize other nodes
    init_msg = {
        "role": "",  # Role name
        "params": {},  # State dict
        "model_config": GPTConfig(),
        "n_nodes": 0,
        "prev_node": {},  # From .json
        "next_node": {},  # From .json
        "n_layers": 0,  # Number of transformer layers
    }

    def __init__(
        self,
        ckpt_path: str,
        nodes_info_path: Union[str, None] = None,
        **setup,
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

        # Override global constants
        if "verb" in setup:
            global VERB
            VERB = bool(setup["verb"])
        if "plots" in setup:
            global PLOTS
            PLOTS = bool(setup["plots"])

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
        self.n_intermediate = len(self.nodes_info["nodes"]["intermediate"])
        self.n_total_nodes = 2 + self.n_intermediate
        try:
            self.model_ckpt = torch.load(ckpt_path, map_location=DEVICE)
        except:
            # It may be that the model does not fit all in the VRAM
            if VERB:
                print("Loading full model on RAM - not enough VRAM")
            logger_wp.warn("Loading full model on RAM - not enough VRAM")
            self.model_ckpt = torch.load(ckpt_path, map_location="cpu")

        # Extract state dict
        self.complete_model = self.model_ckpt["model"]  # State dict

        # Remove problematic keys
        # NOTE: this shouldn't happen anymore (it was a problem in nanoGPT)
        unwanted_prefix = "_orig_mod."
        for k, _ in list(self.complete_model.items()):
            if k.startswith(unwanted_prefix):
                self.complete_model[
                    k[len(unwanted_prefix) :]
                ] = self.complete_model.pop(k)

        # Split model - NOTE: the function removes the elements from
        # self.complete_model, saving memory (no duplicate values)
        self.model_chunks, self.layers_info = split_parameters(
            model_params=self.complete_model, n_nodes=self.n_total_nodes
        )
        assert (
            len(self.complete_model) == 0
        ), "Something went wrong when splitting model - leftover parameters!"

        self.n_layers_tot = (
            self.layers_info["N_LAYERS_START"]
            + self.layers_info["N_LAYERS_FINISH"]
            + self.n_intermediate * self.layers_info["N_LAYERS_INTERM"]
        )

        global MODEL_TYPE
        MODEL_TYPE = f"{self.n_layers_tot}layers"

        # Extract tokenizer metadata information and check it exists
        if (
            "config" in self.model_ckpt
            and "DATASET" in self.model_ckpt["config"]
        ):
            dataset_name = os.path.basename(
                os.path.normpath(self.model_ckpt["config"]["DATASET"])
            )
            self.tok_meta_path = os.path.join(
                script_dir,
                "..",
                "data",
                dataset_name,
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
        starter_config["n_nodes"] = self.n_total_nodes
        starter_config["prev_node"] = self.nodes_info["nodes"]["finisher"]
        starter_config["n_layers"] = self.layers_info["N_LAYERS_START"]
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
            curr_msg["n_nodes"] = self.n_total_nodes

            curr_msg["prev_node"] = prev
            curr_msg["next_node"] = next

            curr_msg["n_layers"] = self.layers_info["N_LAYERS_INTERM"]

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
                logger_wp.error("Failed to initialize node!")
                return out

            if VERB:
                print("> Success!")
            logger_wp.info("Node was initialized successfully")

        # Finisher config - can use next/prev from last loop iteration
        curr_msg = self.init_msg.copy()
        curr_msg["role"] = "finisher"
        curr_msg["model_config"] = self.model_config.asdict()
        curr_msg["params"] = serialize_params(self.model_chunks["finisher"])
        curr_msg["n_nodes"] = self.n_total_nodes

        curr_msg["prev_node"] = prev
        curr_msg["next_node"] = next

        curr_msg["n_layers"] = self.layers_info["N_LAYERS_FINISH"]

        target_addr = self.nodes_info["nodes"]["finisher"]["addr"]
        target_port = self.nodes_info["nodes"]["finisher"]["communication"][
            "port"
        ]
        addr = f"http://{target_addr}:{target_port}/init"
        if VERB:
            print(f"Initializing finisher node ({addr})")
        logger_wp.info(f"Initializing finisher node ({addr})")

        out *= self.request_to_node("post", addr, curr_msg)

        if out:
            if VERB:
                print("> Success!")
            logger_wp.info("Node was initialized successfully!")
        elif not out:
            if VERB:
                print("> Failed!")
            logger_wp.error("Failed to initialize finisher node")

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
            logger_wp.debug(
                f"Successful {req_type} request sent to {addr} - code {ret.status_code}"
            )
            if ret.status_code == 413:
                raise ConnectionError(
                    f"Max payload for {req_type} was exceeded!"
                )
        except:
            logger_wp.warning(
                f"Unable to submit {req_type} request sent to {addr}"
            )
            n_ret += 1
        while (
            ret is None or ret.status_code != 200
        ) and n_ret < max_n_requests:
            if VERB:
                print(
                    f"Unable to reach node ({addr}) - retrying in 2s ({n_ret}/{max_n_requests})"
                )
            time.sleep(2)
            try:
                ret = req_func(addr, json=content)
                logger_wp.debug(
                    f"Successful {req_type} request sent to {addr} - code {ret.status_code}"
                )
            except:
                logger_wp.warning(
                    f"Unable to submit {req_type} request sent to {addr}"
                )
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
