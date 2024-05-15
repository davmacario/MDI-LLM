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
from sub.config import DTYPE, HEADERLENGTH, N_LAYERS_NODES, TEMPERATURE, TOP_K
from sub.model import KVCache, sample
from sub.submodels import SecondaryNode, StarterNode
from sub.typing import FileType
from sub.utils import (count_transformer_blocks, load_sd, loading_bar,
                       plot_tokens_per_time)

from llama.sub.prompts import PromptStyle, has_prompt_style, load_prompt_style

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
        chunk_path: Optional[FileType] = None,
        tokenizer_dir: Optional[FileType] = None,
        model_device: Optional[str] = "cpu",
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
            node_type: string indicating the node type/role ("starter" or "secondary")
                - to indicate a specific secondary node, the node type should be
                "secondary:n" where n is the zero-based index
            *
            model_config: Config object
            chunk_path: path of the model chunk - for the starter node, it should be
                provided always [this assumes the model has been partitioned already by
                the wrapper class GPTDistr]
            tokenizer_dir: directory containing the tokenizer config files
            model_device: device where to load the model chunk; can be omitted if
                specified in the node_config (key "device")
            [**kwargs: support for 'verb' and 'plots' bool values]
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
            assert chunk_path is not None, "Missing path to the model chunk"
            assert model_config is not None, "Missing model Config"
            assert tokenizer_dir is not None, "Missing tokenizer directory"

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
            self.torch_model_device = torch.device(self.model_device)
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
            # Check n. of detected parameters are consistent with chunk (count_transformer_blocks)
            n_layers_detect = count_transformer_blocks(model_params)
            if self.n_layers_local != n_layers_detect:
                raise ValueError(
                    f"The number of detected transformer blocks ({n_layers_detect}) is "
                    f"different from the expected one {self.n_layers_local}; please "
                    "re-run the model partition or check the configuration!"
                )
            self._init_model(model_params, self.n_layers_local)
            # Clear memory of model_params
            del model_params
            model_params = None
            gc.collect()

            # Initialize tokenizer
            self._load_tokenizer(tokenizer_dir)

        else:
            # model_config and chunk_path may be absent!
            self.model_config = model_config  # Should be None
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
                elif "secondary" in node_config and len(node_config["secondary"]) == 1:
                    self.role = "secondary:0"
                    secondary_index = 0
                else:
                    raise RuntimeError(
                        "Unable to infer which secondary node this is - please specify "
                        "the role as 'secondary:n' where 'n' is the index"
                    )
            else:
                secondary_index = int(split_node_type[1])
                self.role = f"secondary:{secondary_index}"

            # For secondary nodes, `node_config` can also be just the specific node
            self.own_config = (
                node_config
                if "secondary" not in node_config
                else node_config["secondary"][secondary_index]
            )
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
            self.torch_model_device = torch.device(self.model_device)

            self._running_thread = threading.Thread()  # Placeholder FIXME
            # NOTE: the model will be initialized once config info is received (POST)

        # Init own info
        self.own_addr = self.own_config["addr"]
        self.own_comm_port = self.own_config["communication"]["port"]
        self.inference_port_in = self.own_config["inference"]["port_in"]
        self.inference_port_out = self.own_config["inference"]["port_out"]

        self.start_webserv()

    # ---------------------------------------------------------------------------------
    def start_webserv(self):
        """
        Launch the web server.
        """
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

    def stop_webserv(self):
        cp.engine.stop()

    # ---------------------------------------------------------------------------------

    def launch_starter(
        self, n_samples: int, max_tokens: int, prompt: str
    ) -> Tuple[List[str], int]:
        """
        Launch processing thread in starter node.

        This method should be called once all the nodes in the network have been
        initialized.

        Args:
            n_samples: number of produced samples (pieces of text)
            max_tokens: max. number of tokens per sample
            prompt: prompt from command line argument (can be prompt itself or FILE:...)

        Returns:
            generated text samples (list of strings)
            generation time (total)
        """
        if self.role != "starter":
            raise ValueError(f"Cannot run `launch_starter` for node type {self.role}")
        metrics_dict = {}
        self.inference_thread = threading.Thread(
            target=self.start_inference,
            args=(n_samples,),
            kwargs={
                "max_new_tokens": max_tokens,
                "prompt": prompt,
                "metrics": metrics_dict,
            },
        )
        self.inference_thread.start()
        self.inference_thread.join()
        return metrics_dict["gen_text"], metrics_dict["gen_time"]

    def start_inference(
        self,
        n_samples: int,
        *,
        max_new_tokens: Optional[int] = None,
        prompt: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ):
        """
        This method is meant to be ran as an independent thread.

        Perform normal operation (open sockets, wait for communication from previous
        node and forward activations to next one).

        In starter nodes, the function launches the operation by creating sockets to the
        nodes and initializing the sample vectors.
        Starter nodes are the only ones for which the arguments should not be None.
        The loop, for starter nodes, is not infinite, as they should know how many
        tokens to generate.

        This function launches an infinite loop on a separate thread in secondary
        nodes, interrupted by the receival of a special message (PUT) over the
        communication channel that triggers a change in a class attribute.
        Non-starter nodes do not know how long the generation will take, hence they need
        to be stopped "externally" by the starter node once the generation is complete.

        Args:
            n_samples: number of samples to be generated (i.e., independent pieces of
                text)
            max_new_tokens (starter only): maximum number of tokens per generated
                sample
            prompt (starter only): string containing the prompt or "FILE:<filename.txt>"
            metrics (starter only): dict where the metrics will be inserted (keys:
                "gen_text" and "gen_time")
        """
        assert self.sock_to_prev is None and self.sock_to_next is None
        assert self.next_node is not None and self.prev_node is not None
        assert self.model_config is not None and self.model is not None

        # Configuration for all nodes
        self._create_sockets()
        assert self.sock_to_prev is not None and self.sock_to_next is not None

        # Differentiate between different types
        if self.node_type == "starter":
            assert max_new_tokens is not None

            self.n_samples = n_samples
            self.running = True
            self._launch_queue_threads()

            if VERB:
                print(
                    f"[INFO] Starting generation loop - {n_samples} samples, {max_new_tokens} tokens each"
                )
            logger_wp.info("Starting generation loop")

            out_text, gen_time = self._starter_loop(n_samples, max_new_tokens, prompt)

            if metrics is not None:
                # NOTE: this allows to return values even if this method is on a
                # separate thread! Just read from this object after `join`
                metrics["gen_text"] = out_text
                metrics["gen_time"] = gen_time
        else:
            # Secondary node
            self.running = True
            self._launch_queue_threads()

            if VERB:
                print("[INFO] Starting generation loop")
            logger_wp.info("Starting generation loop")
            self._secondary_loop()

    def shutdown(self) -> int:
        """
        Turn off the node - stop server, close sockets and stop thread.

        Returns:
            1 upon success, 0 otherwise (exception gets raised)
        """
        if VERB:
            print("[INFO] Shutting down")

        try:
            time.sleep(2)
            self.running = False  # Redundant, but ok
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
            self.stop_webserv()
            if VERB:
                print("Closing application")
            return 1
        except:
            return 0

    # ----- Inference message transmission --------------------------------------------
    def _launch_queue_threads(self):
        """
        Launch the input and output queue threads;
        This method is called by `start_inference()`.
        """
        if VERB:
            print("[INFO] Starting queue threads")
        logger_wp.info("Starting queue threads")
        self.in_queue_thread = threading.Thread(
            target=self._fill_input_queue, daemon=True
        )
        self.in_queue_thread.start()

        self.out_queue_thread = threading.Thread(
            target=self._empty_output_queue, daemon=True
        )
        self.out_queue_thread.start()

    def _recv_from_prev(self, size: int) -> bytes:
        """
        Receive a message of the specified size from the previous node.

        Remark: the size specified in socket.recv(...) is the MAX size that will be read
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
                f"[INFO] Started listening on port {self.own_config['inference']['port_in']}"
            )
        self.sock_to_prev.listen(1)

        self.sock_to_prev_prop = self.sock_to_prev.accept()
        logger_wp.info("Created socket to previous node")

        if VERB:
            print("-> Done!                     ")

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
                        self.own_config["addr"],
                        self.own_config["inference"]["port_in"],
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
                f"Unable to bind to ({self.own_config['addr']}, {self.own_config['inference']['port_out']})"
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
                        self.own_config["addr"],
                        self.own_config["inference"]["port_out"],
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
                f"Unable to create client socket at ({self.own_config['addr']}, {self.own_config['inference']['port_in']})"
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
            msg = self._recv_from_prev(HEADERLENGTH)

            # Extract message length from the header
            msg_len = int(msg[:HEADERLENGTH])
            _n_recv_msg += 1

            # Read payload (exact size - this is important)
            msg_payload = self._recv_from_prev(msg_len)
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
                self._send_to_next(tx_msg)
                self.running = False
                break

            self._send_to_next(tx_msg)

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
        assert self.model_device is not None, "No device was specified"

        if VERB:
            print("Initializing local model")

        Model_class = StarterNode if "starter" in self.node_type else SecondaryNode
        self.model = Model_class(self.model_config, n_transf_layers)
        self.model.load_weights(model_parameters)
        self.model = self.model.to(self.torch_model_device)
        # NOTE: need to init_kv_cache once the number of samples (batch size) is known

    def _load_tokenizer(self, tokenizer_dir: FileType):
        """
        Load the tokenizer information and prompt style definition from the specified
        path.
        The tokenizer object will be stored in `self.tok`, while the prompt style will
        be stored in `self.prompt_style`.

        Args:
            tokenizer_dir: path to the directory containing the tokenizer config files
        Returns:
            True if the operation was successful
        """
        if VERB:
            print("Loading tokenizer", end="")
        self.tok = Tokenizer(tokenizer_dir)
        tok_dir_path = (
            Path(tokenizer_dir) if isinstance(tokenizer_dir, str) else tokenizer_dir
        )
        if not has_prompt_style(tok_dir_path):
            assert self.model_config is not None
            self.prompt_style = PromptStyle.from_config(self.model_config)
        else:
            self.prompt_style = load_prompt_style(tok_dir_path)
        if VERB:
            print("Tokenizer and prompt style have been loaded!")

    def _starter_loop(
        self, n_samples: int, max_new_tokens: int, prompt: Optional[str] = None
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
        assert self.model_device is not None

        if n_samples < 1:
            raise ValueError("Cannot generate less than 1 sample!")
        elif n_samples < self.n_nodes:
            warnings.warn(
                f"Generating less samples ({n_samples}) than nodes ({self.n_nodes}) will not be efficient!"
            )
        self.n_samples = n_samples

        if VERB:
            print("Initializing model cache")
        # TODO: figure out KV cache usage - it is initialized to the batch size, but we
        # need to use only a specific one of the n_samples dimensions for each sample

        if "cuda" in self.model_device:
            device_type = "cuda"
        elif "mps" in self.model_device:
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

        start_styled = [self.prompt_style.apply(s) for s in start]

        assert len(start) == n_samples

        idx = [
            self.tok.encode(txt, device=self.torch_model_device).view(1, -1)
            for txt in start_styled
        ]

        # IDEA: use set_kv_cache for 1st sample, others are init directly
        # This is needed because set_kv_cache also initializes the mask, which is the
        # same regardless of samples (matrix of 1s and 0s)
        self.model.set_kv_cache(batch_size=1, device=self.torch_model_device)
        # Prompt size T and input_pos are now lists of n_samples elements
        T_i: List[int] = []  # Contains size of context of each prompt
        input_pos: List[torch.Tensor] = []  # Will contain the input pos. of all samples
        kvcaches: List[List[KVCache]] = []
        for i in range(n_samples):
            T_i.append(idx[i].size(1))
            input_pos.append(torch.arange(0, T_i[i], device=self.torch_model_device))

            kvc_sublist: List[KVCache] = []
            for block in self.model.transformer.h:
                # Build kv cache individually for each attn layer
                if i == 0:
                    # Copy the already init KVCaches of the blocks
                    kvc_sublist.append(block.attn.kv_cache)
                else:
                    # Create caches with block.attn.build_kv_cache()
                    kvc_sublist.append(
                        block.attn.build_kv_cache(
                            batch_size=1,
                            max_seq_length=self.model.max_seq_length,
                            rope_cache_length=self.model.cos.size(-1),
                            device=self.torch_model_device,
                            dtype=DTYPE,
                        )
                    )
            kvcaches.append(kvc_sublist)

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
                    # Identify sample
                    sample_id = k % n_samples

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

                        idx_from_fin = in_msg["data"].to(self.model_device)

                        # NOTE: no KV caching here - no need to pass input_pos
                        logits = self.model(idx_from_fin, first_pass=False)
                        idx_next = sample(
                            logits, temperature=self.temperature, top_k=self.top_k
                        )
                        idx[sample_id] = torch.cat((idx[sample_id], idx_next), dim=1)
                        input_pos[sample_id] = input_pos[sample_id][-1:].add_(1)

                    # Send to next iff not at the last token
                    if k < (n_samples * (max_new_tokens - 1)):
                        # Crop to block size
                        idx_cond = (
                            idx[sample_id]
                            if idx[sample_id].size(1) <= self.model_config.block_size
                            else idx[sample_id][:, -self.model_config.block_size :]
                        )
                        # NOTE: Swap KVCache for correct sample
                        curr_kvcache = kvcaches[sample_id]
                        for ind_b, block in enumerate(self.model.transformer.h):
                            block.attn.kv_cache = curr_kvcache[ind_b]

                        # Forward in local model (first piece)
                        idx_cond = self.model(idx_cond, input_pos[sample_id])

                        # Send message
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

    def _secondary_loop(self):
        """
        Execution loop for non-starter nodes. This method must be used as the target of
        a thread that is launched once the node has been correctly initialized.

        The execution will be stopped once a PUT request is made to /stop.
        """
        assert self.sock_to_prev is not None and self.sock_to_next is not None
        assert self.model is not None and self.model_config is not None
        assert self.n_samples is not None
        assert self.model_device is not None

        # Should be overrided by kwargs
        if "cuda" in self.model_device:
            device_type = "cuda"
        elif "mps" in self.model_device:
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

        # IDEA: use set_kv_cache for 1st sample, others are init directly
        # This is needed because set_kv_cache also initializes the mask, which is the
        # same regardless of samples (matrix of 1s and 0s)
        self.model.set_kv_cache(batch_size=1, device=self.torch_model_device)
        kvcaches: List[List[KVCache]] = []
        for i in range(self.n_samples):
            kvc_sublist: List[KVCache] = []
            for block in self.model.transformer.h:
                # Build kv cache individually for each attn layer
                if i == 0:
                    # Copy the already init KVCaches of the blocks
                    kvc_sublist.append(block.attn.kv_cache)
                else:
                    # Create caches with block.attn.build_kv_cache()
                    kvc_sublist.append(
                        block.attn.build_kv_cache(
                            batch_size=1,
                            max_seq_length=self.model.max_seq_length,
                            rope_cache_length=self.model.cos.size(-1),
                            device=self.torch_model_device,
                            dtype=DTYPE,
                        )
                    )
            kvcaches.append(kvc_sublist)

        self.model.eval()

        # Prompt size T and input_pos are now lists of n_samples elements - will be init
        # once the generation starts
        T_i: List[int] = []  # Contains size of context of each prompt
        input_pos: List[torch.Tensor] = []  # Will contain the input pos. of all samples

        loopsigns = ["|", "/", "-", "\\"]
        iter = 0
        exp_ind = 0  # Expected sample index from previous
        is_first_iter = True  # True for the first n_samples iters
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
                        print("[DEBUG] Received stopping message over socket")

                    if self.running:
                        sample_id = in_msg["sample_index"]
                        assert (
                            exp_ind == sample_id
                        ), f"Expected sample index {exp_ind}, received {sample_id}"
                        exp_ind = (sample_id + 1) % self.n_samples

                        if iter >= self.n_samples:
                            is_first_iter = False

                        idx = in_msg["data"].to(self.torch_model_device)
                        if is_first_iter:
                            # Initialization of the input_pos
                            T_i.append(idx.size(1))
                            input_pos.append(
                                torch.arange(
                                    0, T_i[sample_id], device=self.torch_model_device
                                )
                            )

                        print(f"> Generating {loopsigns[iter % 4]}", end="\r")
                        # Swap KVCache
                        curr_kvcache = kvcaches[sample_id]
                        for ind_b, block in enumerate(self.model.transformer.h):
                            block.attn.kv_cache = curr_kvcache[ind_b]

                        # Forward pass
                        outs = self.model(idx, input_pos=input_pos[sample_id])

                        # Build msg
                        out_msg = self._build_msg(outs, sample_id)
                        # Send to next
                        self.out_message_queue.append(out_msg)
                        self.out_queue_not_empty.set()

                        input_pos[sample_id] = input_pos[sample_id][-1:].add_(1)
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
            Message fields:
                role ("secondary:n" - does NOT overwrite the previous node given at init)
                prev_node (as in configuration json file)
                next_node (as in configuration json file)
                model_config (serialized with Config.asdict())
                n_nodes (number of total network nodes)
                [params] (model parameters - needed if no chunk path was passed)
                n_samples (number of produced samples)
        """
        if (
            self.node_type is None or "secondary" in self.node_type
        ) and self.model is None:  # Only for non-init nodes
            if len(path) > 0 and path[0] == "init":
                assert not self.running
                init_msg = pickle.loads(cp.request.body.read())
                if self.node_type is None:
                    self.node_type = init_msg["role"]
                self.prev_node = init_msg["prev_node"]
                self.next_node = init_msg["next_node"]
                # Assume model config is not initialized
                self.model_config = Config(**init_msg["model_config"])
                self.n_nodes = init_msg["n_nodes"]
                self.n_layers_local = N_LAYERS_NODES[self.n_nodes][
                    self.model_config.n_layer
                ]["N_LAYERS_SECONDARY"]

                if "params" in init_msg:
                    if VERB:
                        print("Received parameters from starter")
                    chunk_sd = init_msg["params"]
                else:
                    if self.model_path is None:
                        raise RuntimeError(
                            "The received message did not contain the model parameters "
                            "- please specify a model chunk path when initializing "
                            "GPTServer object"
                        )
                    if VERB:
                        print("Loading parameters from disk")
                    chunk_sd = load_sd(self.model_path)

                # Check
                n_layers_detect = count_transformer_blocks(chunk_sd)
                if self.n_layers_local != n_layers_detect:
                    raise ValueError(
                        f"The number of detected transformer blocks ({n_layers_detect}) is "
                        f"different from the expected one {self.n_layers_local}; please "
                        "re-run the model partition or check the configuration!"
                    )

                self._init_model(chunk_sd, self.n_layers_local)
                # Clear memory of model_params
                del chunk_sd
                chunk_sd = None
                gc.collect()

                self.n_samples = init_msg["n_samples"]

                if VERB:
                    print(f"{self.n_nodes} Nodes, generating {self.n_samples} samples")

                if VERB:
                    print(f"[INFO] Starting operation - {self.node_type} node")
                # FIXME: review threads
                logger_wp.info("Received initialization information!")
                self._running_thread = threading.Thread(
                    target=self.start_inference, daemon=True
                )
                self._running_thread.start()
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found")
        elif self.model is not None:
            raise cp.HTTPError(
                403,
                f"Failed to configure node - the model was already initialized: {self.node_type}",
            )
        else:
            raise cp.HTTPError(403, "Unable to initialize node!")

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
