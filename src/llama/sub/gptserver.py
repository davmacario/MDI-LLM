import gc
import json
import logging
import os
import pickle
import socket
import threading
import time
import warnings
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cherrypy as cp
import torch
import torch.nn.functional as F
from sub import Config, Tokenizer, get_user_prompt
from sub.config import HEADERLENGTH, N_LAYERS_NODES, TEMPERATURE, TOP_K
from sub.submodels import SecondaryNode, StarterNode
from sub.utils import load_sd, loading_bar, plot_tokens_per_time

# -------------------------------------------------------------------------------------

script_dir = Path(os.path.dirname(__file__))

# TODO: logger
logger_wp = logging.getLogger("model_dist")
logger_wp.setLevel(logging.ERROR)

MODEL_TYPE = ""
VERB = False


class GPTServer:
    """
    Communication server - Cherrypy-based webserver used for exchanging
    (receiving) setup and control information
    """

    exposed = True
    model: Optional[Union[StarterNode, SecondaryNode]] = None
    next_node: Optional[Dict] = None
    prev_node: Optional[Dict] = None
    model_params: Optional[Dict] = None
    model_config: Optional[Config] = None

    # True iff the model has been initialized and it is ready to perform inference.
    running: bool = False

    sock_to_prev: Optional[socket.socket] = None
    sock_to_prev_prop: Tuple = ()  # NOTE: used now
    sock_to_next: Optional[socket.socket] = None
    sock_to_next_prop: Tuple = ()  # NOTE: not used!

    n_samples: Optional[int] = None

    stop_msg = {"stop": True}

    # Input message queue
    in_message_queue = deque([])
    in_queue_not_empty = threading.Event()  # Replaces busy waiting
    in_queue_not_empty.clear()
    # Output message queue
    out_message_queue = deque([])
    out_queue_not_empty = threading.Event()
    out_queue_not_empty.clear()

    # Some model configs:
    top_k = TOP_K
    temperature = TEMPERATURE

    # Stats - n. tokens/time (tuples)
    tok_time: List = []

    def __init__(
        self,
        node_config: Dict,
        node_type: str,
        *,
        model_config: Optional[Config] = None,
        chunk_path: Optional[Union[str, Path]] = None,
        model_device: Optional[Union[str, torch.device]] = "cpu",
        **kwargs,
    ):
        """
        Initialize GPTServer object.

        This object will control a specific model (Starter/Secondary), allowing to pass
        on the information in the chain while performing inference.

        The couple 'node_config' & 'node_type' should be enough to uniquely identify
        the node.

        Args:
            node_config: node configuration information (from .json file)
            node_type: string indicating the node type - to indicate a specific
                secondary node, the node type should be "secondary:n" where n is the
                zero-based index
            *
            model_config: Config object
            starter_config: extra arguments required for the starter node; expected
                keys:
                - params: model parameters (state dict) for starter node
                - model_config: GPTConfig object
                - next_node: info about next node
                - prev_node: info about previous node
                - tok_metadata_path: path of the tokenizer metadata (for
                    CharacterTokenizer)
            chunk_path: path of the model (chunk) - for the starter node, it should be
                provided always [this assumes the model has been partitioned already by
                the wrapper class GPTDistr]
        """
        # NOTE: this implementation supports running 1 node only
        # Override global constants with kwargs
        if "verb" in kwargs:
            global VERB
            VERB = bool(kwargs["verb"])
            print(f"Overriding 'verb': {VERB}")
        if "plots" in kwargs:
            global PLOTS
            PLOTS = bool(kwargs["plots"])
            print(f"Overriding 'plots': {PLOTS}")

        self.node_type = node_type
        self.node_config = node_config

        if "starter" in node_type:
            assert chunk_path is not None
            assert model_config is not None

            if isinstance(chunk_path, str):
                self.model_path = Path(chunk_path)
            else:
                self.model_path = chunk_path

            # The node_config for the starter is the whole json! It should know the
            # other nodes in the network to initialize them
            self.role = "starter"
            self.own_config = node_config["starter"]

            # Possibly get device info if found in config file
            try:
                self.model_device = (
                    model_device
                    if "device" not in self.own_config
                    else self.own_config["device"]
                )
            except KeyError:
                raise ValueError(
                    "Missing model device information - either specify it in `node_config`"
                    " or pass argument `model_device` to this function"
                )
            self.n_nodes = 1 + len(node_config["secondary"])
            self.next_node = None if self.n_nodes == 1 else node_config["secondary"][0]
            self.prev_node = None if self.n_nodes == 1 else node_config["secondary"][-1]

            # Extract model params (to cpu)
            self.model_config = model_config
            self.n_layers_local = (
                self.model_config.n_layer
                if self.n_nodes == 1
                else N_LAYERS_NODES[self.n_nodes][self.model_config.n_layer][
                    "N_LAYERS_START"
                ]
            )
            # Load chunk
            model_params = load_sd(self.model_path)
            # TODO: check n. of detected parameters are consistent with chunk
            # Will empty "model_params"
            self._init_model(model_params, self.n_layers_local)
        else:
            # model_config and chunk_path may be absent!
            self.model_config = model_config
            if isinstance(chunk_path, str):
                self.model_path = Path(chunk_path)
            else:
                self.model_path = chunk_path  # May be None

            # Parse role name to get right node config
            split_node_type = node_type.split(":")
            if len(split_node_type) == 1:
                if len(node_config["secondary"]) > 1:
                    raise ValueError(
                        "Need to specify which of the secondary nodes this is"
                    )
                elif len(node_config["secondary"]) == 1:
                    self.role = "secondary:0"
                    secondary_index = 0
                else:
                    raise RuntimeError(
                        "No secondary nodes have been specified in the configuration file!"
                    )
            else:
                secondary_index = int(split_node_type[1])
                self.role = f"secondary:{secondary_index}"

            self.own_config = node_config["secondary"][secondary_index]
            self.starter_addr = self.own_config["communication"]["starter_addr"]
            # Possibly get device info if found in config file
            try:
                self.model_device = (
                    model_device
                    if "device" not in self.own_config
                    else self.own_config["device"]
                )
            except KeyError:
                raise ValueError(
                    "Missing model device information - either specify it in `node_config`"
                    " or pass argument `model_device` to this function"
                )

            self._running_thread = threading.Thread()  # Placeholder FIXME
            # NOTE: the model will be initialized once config info is received (POST)

        # Init own info
        self.own_addr = self.own_config["addr"]
        self.own_comm_port = self.own_config["communication"]["port"]
        self.inference_port_in = self.own_config["inference"]["port_in"]
        self.inference_port_out = self.own_config["inference"]["port_out"]

        # Launch web server
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

    # ---------------------------------------------------------------------------------

    def start(
        self,
        n_samples: Union[None, int] = None,
        max_new_tokens: Union[None, int] = None,
        prompt: Union[None, str] = None,
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
            n_samples: number of samples to be generated (i.e., independent pieces of
                text)
            max_new_tokens: ONLY FOR STARTER - maximum number of tokens per generated
                sample
            prompt: (STARTER ONLY) - string containing the prompt or
                "FILE:<filename.txt>"

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
            assert max_new_tokens is not None and n_samples is not None

            self.n_samples = n_samples

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

            # Input queue
            self.in_queue_thread = threading.Thread(
                target=self._fill_input_queue, daemon=True
            )
            self.in_queue_thread.start()
            # Output queue
            self.out_queue_thread = threading.Thread(
                target=self._empty_output_queue, daemon=True
            )
            self.out_queue_thread.start()

            if VERB:
                print(
                    f"[INFO] Starting generation loop - {n_samples} samples, {max_new_tokens} tokens each"
                )
            logger_wp.info("Starting generation loop")

            out_text, gen_time = self._starter_loop(n_samples, max_new_tokens, prompt)

            return out_text, gen_time
        else:
            self.running = True
            if VERB:
                print("[INFO] Starting queue thread")
            logger_wp.info("Starting queue thread")
            self.in_queue_thread = threading.Thread(
                target=self._fill_input_queue, daemon=True
            )
            self.in_queue_thread.start()

            self.out_queue_thread = threading.Thread(
                target=self._empty_output_queue, daemon=True
            )
            self.out_queue_thread.start()

            if VERB:
                print("[INFO] Starting generation loop")
            logger_wp.info("Starting generation loop")
            self._node_loop()

    def shutdown(self) -> int:
        """
        Turn off the node - stop server, close sockets and stop thread.

        Returns:
            1 upon success, 0 otherwise
        """
        if VERB:
            print("[INFO] Shutting down")

        try:
            time.sleep(2)
            self.running = False  # Redundant
            if self.node_type != "starter":
                if VERB:
                    print("Stopping main thread")
                self._running_thread.join()
            if VERB:
                print("Stopping input queue thread")
            self.in_queue_thread.join()
            if VERB:
                print("Stopping output queue thread")
            self.out_queue_thread.join()
            if VERB:
                print("Stopping input and output sockets")
            self.sock_to_prev_prop[0].close()
            self.sock_to_prev.close()
            self.sock_to_next.close()
            if VERB:
                print("Stopping HTTP server")
            cp.engine.exit()
            if VERB:
                print("Closing application")
            return 1
        except:
            return 0

    # ----- Inference message transmission --------------------------------------------

    def _recv_from_prev(self, size: int) -> bytes:
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

    def _send_to_next(self, data: Any):
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

    def _create_sockets(self):
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
            print(
                f"[INFO] Started listening on port {self.node_config['inference']['port_in']}"
            )
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

    def _fill_input_queue(self):
        """
        This method has the goal of managing incoming messages from previous nodes in
        the chain.
        As a message is received, its contents are stored in the message queue
        (`self.in_message_queue`).
        This allows to store locally each of the received messages, in order.
        The order is crucial for the correct functioning of the program (pipelining).

        This method loops infinitely and constantly waits for incoming messages.
        For this reason, it is ran on a separate thread, and it is stopped when the main
        thread, running the processing function, finishes.
        """
        assert self.sock_to_prev is not None and self.sock_to_prev_prop != ()

        if len(self.in_message_queue) < 1:
            self.in_queue_not_empty.clear()

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
                self.in_message_queue.append(data)
                self.in_queue_not_empty.set()
                self.running = False
            else:  # Not here if stopping message is received
                self.in_message_queue.append(data)
                self.in_queue_not_empty.set()

        if VERB:
            print("Input queue thread stopped")

    def _empty_output_queue(self):
        """
        Handle transmission of messages in the output queue. As messages are placed in
        the output queue by the execution loops, it will use the output socket to send
        them to the next node in the chain.

        This method should run on its separate thread.
        """
        assert self.sock_to_next is not None

        while self.running:
            # Wait for message in queue
            assert self.out_queue_not_empty.wait()

            tx_msg = self.out_message_queue.popleft()
            if len(self.out_message_queue) < 1:
                self.out_queue_not_empty.clear()

            if "stop" in tx_msg and tx_msg["stop"]:
                self.send_to_next(tx_msg)
                self.running = False
                break

            self.send_to_next(tx_msg)

        if VERB:
            print("Output queue thread stopped")

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

    # ----- Private -------------------------------------------------------------------

    def _init_model(self, model_parameters: Dict[str, Any], n_transf_layers: int):
        """
        Initialize the node's model chunk and move it to the target device
        (self.model_device).

        Args:
            model_parameters: state dict containing the weights - will be emptied
            n_transf_layers: number of transformer layers of the local model; required
                for initializing the submodel.
        """
        assert self.model_config is not None, "No model configuration was found!"
        assert self.model is None, "The model was already initialized!"

        if "starter" in self.node_type:
            self.model = StarterNode(self.model_config, n_transf_layers)
        elif "secondary" in self.node_type:
            self.model = SecondaryNode(self.model_config, n_transf_layers)
        else:
            raise ValueError(f"Unrecognized node type {self.node_type}")

        self.model.load_weights(model_parameters)
        self.model = self.model.to(self.model_device)

        # Clear memory of model_params
        del self.model_params
        self.model_params = None
        gc.collect()

    def _load_tokenizer(
        self,
    ) -> Tokenizer:
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

    def _starter_loop(
        self, n_samples: int, max_new_tokens: int, prompt: Union[str, None] = None
    ) -> Tuple[List[str], float]:
        """
        Generation loop for the starter node only.
        This loop has a finite duration, as the starter knows what is the length of the
        samples to be generated.

        Args:
            n_samples: number of produced samples
            max_new_tokens: maximum number of tokens
            prompt: either the prompt itself or a string of the type "FILE:<prompt.txt>"
                containing each prompt as a separate paragraph

        Returns:
            list containing the `n_nodes` generated samples
            total generation time in seconds
        """
        assert self.model_config is not None and self.model is not None

        if n_samples is None:
            n_samples = self.n_nodes
        elif n_samples < 1:
            raise ValueError("Cannot generate less than 1 sample!")
        elif n_samples < self.n_nodes:
            warnings.warn(
                f"Generating less samples ({n_samples}) than nodes ({self.n_nodes}) will not be efficient!"
            )

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
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # Encode starting sequence - with prompt support
        if prompt is None:
            start = ["\n"] * n_samples
        else:
            start = get_user_prompt(prompt, n_samples, verb=VERB)

        assert len(start) == n_samples

        start_ids = [self.tok_encode(txt) for txt in start]
        idx = [
            torch.tensor(start_txt, dtype=torch.long, device=self.device)[None, ...]
            for start_txt in start_ids
        ]

        self.model.eval()
        start_time = time.time()
        if PLOTS:
            self.tok_time.append((0, 0))
        with torch.no_grad():
            with ctx:
                total_iters = max_new_tokens * n_samples
                for k in range(total_iters):
                    logger_wp.info(f"Iter {k}")
                    print(
                        f"Generating: {loading_bar(k, total_iters, 20)} ({k}/{total_iters})",
                        end="\r",
                    )
                    if PLOTS:
                        self.tok_time.append((k, time.time() - start_time))
                    sample_id = k % n_samples  # Identify sample

                    if k >= n_samples:
                        # We are not in the first iteration (k starts from 0)
                        # can start processing messages from last secondary node

                        # Wait for queue to contain msg
                        assert self.in_queue_not_empty.wait()

                        in_msg = self.in_message_queue.popleft()
                        if len(self.in_message_queue) < 1:
                            self.in_queue_not_empty.clear()
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
                    if k < (n_samples * (max_new_tokens - 1)):
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

                        self.out_message_queue.append(out_msg)
                        self.out_queue_not_empty.set()

        tot_time = time.time() - start_time
        if PLOTS:
            self.tok_time.append((total_iters, tot_time))
            # Store plotted points as csv file
            points_file_path = os.path.join(
                script_dir,
                "..",
                "logs",
                "tok-per-time",
                f"tokens_time_samples_mdi_{MODEL_TYPE}_{n_samples}samples_{self.n_nodes}nodes.csv",
            )
            if not os.path.exists(os.path.dirname(points_file_path)):
                os.mkdir(os.path.dirname(points_file_path))
            with open(points_file_path, "w") as f:
                times = [x[1] for x in self.tok_time]
                n_tok = [x[0] for x in self.tok_time]
                for i in range(len(times)):
                    f.write(f"{times[i]},{n_tok[i]}\n")

            plot_tokens_per_time(
                self.tok_time,
                out_path=os.path.join(
                    script_dir, "..", "img", f"tokens_time_mdi_{MODEL_TYPE}.png"
                ),
            )

        # Send stop message to the next (no queue used)
        self.running = False
        if VERB:
            print("[INFO] Sending stopping message over socket  ")
        self.out_message_queue.append(self.stop_msg)
        self.out_queue_not_empty.set()
        logger_wp.info("Generation completed")
        if VERB:
            print("[INFO] Generation completed!                          ")
            print(f"> Total time for generation: {tot_time} s")

        return [self.tok_decode(smp[0].tolist()) for smp in idx], tot_time

    def _node_loop(self):
        """
        Execution loop for non-starter nodes. This method must be used as the target of
        a thread that is launched once the node has been correctly initialized.

        The execution will be stopped once a PUT request is made to /stop.
        """
        assert self.sock_to_prev is not None and self.sock_to_next is not None
        assert self.model is not None and self.model_config is not None
        assert self.n_samples is not None

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
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        self.model.eval()
        loopsigns = ["|", "/", "-", "\\"]
        iter = 0
        exp_ind = 0  # Expected sample index from previous
        with torch.no_grad():
            with ctx:
                while self.running:
                    logger_wp.info(f"Iter {iter}")

                    assert self.in_queue_not_empty.wait()

                    # Extract message from queue
                    in_msg = self.in_message_queue.popleft()
                    if len(self.in_message_queue) <= 0:
                        self.in_queue_not_empty.clear()

                    if "stop" in in_msg:
                        print("[DEBUG] HERE")

                    if self.running:
                        samp_ind = in_msg["sample_index"]
                        assert (
                            exp_ind == samp_ind
                        ), f"Expected sample index {exp_ind}, received {samp_ind}"
                        exp_ind = (samp_ind + 1) % self.n_samples

                        ins = in_msg["data"].to(self.device)
                        print(f"> Generating {loopsigns[iter % 4]}", end="\r")
                        # Forward pass
                        outs = self.model(ins)
                        # Build msg
                        out_msg = self._build_msg(outs, samp_ind)
                        # Send to next
                        self.out_message_queue.append(out_msg)
                        self.out_queue_not_empty.set()
                        iter += 1
                    else:
                        print("> Generation completed!")
                        self.out_message_queue.append(self.stop_msg)
                        self.out_queue_not_empty.set()
                        self.running = False

        if VERB:
            print("Node loop stopped")

    # ----- REST API ------------------------------------------------------------------

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
                    chunk_cont = torch.load(self.chunk_path)
                    # Check compatibility (all keys of chunk_cont should be in init_msg)
                    assert all(
                        [
                            k in init_msg["model_config"]
                            for k in chunk_cont["model_args"]
                        ]
                    ), f"Different settings:\n{chunk_cont['model_args']}\n\n{init_msg['model_config']}"
                    self.model_params = chunk_cont["model"]
                    del chunk_cont
                    gc.collect()
                self.n_nodes = init_msg["n_nodes"]
                self.n_samples = init_msg["n_samples"]
                if VERB:
                    print(f"{self.n_nodes} Nodes, generating {self.n_samples} samples")

                # Set up the node
                self.init_model(init_msg["n_layers"])
                assert (
                    self.model_params is None
                ), "The model parameters were not flushed!"

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
                # self._end_thr.join()  # cannot wait, since thread stops server
                if VERB:
                    print("[INFO] Node stopped!")
                logger_wp.info("Received stopping directive")
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found!")

    def DELETE(self):
        """Not implemented"""
        raise cp.HTTPError(501, "DELETE not implemented!")
