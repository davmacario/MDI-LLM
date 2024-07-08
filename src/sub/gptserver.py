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

import gc
import json
import logging
import os
import pickle
import threading
import time
import warnings
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cherrypy as cp
import torch

from sub.config import DEVICE as DEFAULT_DEVICE
from sub.config import (DTYPE, DTYPE_TORCH_MAPPING, N_LAYERS_NODES,
                        TEMPERATURE, TOP_K)
from sub.connections import InputNodeConnection, OutputNodeConnection
from sub.model import Config, KVCache, sample
from sub.prompts import (PromptStyle, get_user_prompt, has_prompt_style,
                         load_prompt_style)
from sub.submodels import SecondaryNode, StarterNode
from sub.tokenizer import Tokenizer
from sub.typing import FileType
from sub.utils import (catch_loop_errors, count_transformer_blocks,
                       detect_stop_tokens, find_eot, load_sd,
                       plot_tokens_per_time, waiting_animation)

# -------------------------------------------------------------------------------------

script_dir = Path(os.path.dirname(__file__))

# TODO: logger
logger_wp = logging.getLogger("model_dist")
logger_wp.setLevel(logging.ERROR)

VERB = False
PLOTS = False


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
    model_type = None

    # Number of samples that have been requested so far in the current run:
    n_samples: int = 0
    # Map sample ID to n. of iteration - initialized to 0 when new sample is created
    iter_ind: Dict[int, int] = {}

    T_i: Dict[int, int] = {}  # Contains size of context of each prompt
    # Will contain the input pos. of all samples:
    input_pos: Dict[int, torch.Tensor] = {}
    kvcaches: Dict[int, List[KVCache]] = {}

    # Set iff the model has been initialized and it is ready to perform inference.
    running = threading.Event()
    running.clear()

    # Connections
    conn_to_next: Optional[OutputNodeConnection] = None
    conn_to_prev: Optional[InputNodeConnection] = None

    """
    Message format:
    - Sample index: unique ID of the sample; used to select correct cache
    - Data: activation - tensor
    - stop: flag; if set to True, it is used to advertise the end of generation for the
        current sample (by ID)

    NOTE: if set to True, DO NOT PROCESS DATA (should be empty);

    This logic allows to delete caches for completed samples and make samples
    independent.
    """
    msg_format = {"sample_index": 0, "data": None, "stop": False}

    # Input message queue
    in_message_queue = deque([])
    in_queue_not_empty = threading.Event()  # Replaces busy waiting
    in_queue_not_empty.clear()
    # Output message queue
    out_message_queue = deque([])
    out_queue_not_empty = threading.Event()
    out_queue_not_empty.clear()

    # Response queue - used to pass generated responses from loop to HTTP server
    resp_msg_template = {}  # TODO: use Ollama's syntax
    resp: Mapping[int, Dict] = {}  # Will contain the generated message for each sample
    resp_finished: Mapping[int, threading.Event] = (
        {}
    )  # Used to advertise generation end and make server look for message in self.resp

    # Some model configs:
    top_k = TOP_K
    temperature = TEMPERATURE

    # Stats - n. tokens/time (tuples)
    tok_time: List = []

    # Threads
    inference_thread = threading.Thread()
    in_queue_thread = threading.Thread()
    out_queue_thread = threading.Thread()

    def __init__(
        self,
        node_config: Dict,
        node_type: str,
        *,
        model_config: Optional[Config] = None,
        chunk_path: Optional[FileType] = None,
        tokenizer_dir: Optional[FileType] = None,
        model_device: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize GPTServer object.

        This object will control a specific model (Starter/Secondary), allowing to pass
        on the information in the chain while performing inference.

        The couple 'node_config' & 'node_type' should be enough to uniquely identify
        the node.

        NOTE: this class assumes model partition has already been done.

        Args:
            node_config: node configuration information (from .json file)
            node_type: string indicating the node type/role ("starter" or "secondary")
                to indicate a specific secondary node, the node type should be
                "secondary:n" where n is the zero-based index
            *
            model_config: Config object
            chunk_path: path of the model chunk - for the starter node, it should be
                provided always [this assumes the model has been partitioned already by
                the wrapper class GPTDistr]
            tokenizer_dir: directory containing the tokenizer config files
            model_device: device where to load the model chunk; can be omitted if
                specified in the node_config (arg will override it!)
            [**kwargs: support for 'verb' and 'plots' bool values]
        """
        # NOTE: this implementation supports running 1 node only
        # Override global constants with kwargs
        if "verb" in kwargs:
            global VERB
            VERB = bool(kwargs["verb"])
            if VERB:
                print(f"Overriding 'verb': {VERB}")
        if "plots" in kwargs:
            global PLOTS
            PLOTS = bool(kwargs["plots"])
            if VERB:
                print(f"Overriding 'plots': {PLOTS}")
        if "model_type" in kwargs:
            self.model_type = str(kwargs["model_type"])
            if VERB:
                print(f"Overriding model type: {self.model_type}")
        if "model_seq_length" in kwargs:
            self.max_seq_length = kwargs["model_seq_length"]
        else:
            self.max_seq_length = None

        self.compile = False if "compile" not in kwargs else kwargs["compile"]
        self.use_default_dtype = dtype is None
        self.dtype = dtype if dtype else DTYPE  # Default
        self.ptdtype = DTYPE_TORCH_MAPPING[self.dtype]
        if self.dtype == "bfloat16" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError(
                "Specified bfloat16, but the host does not support this format"
            )

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

            if isinstance(tokenizer_dir, str):
                self.tokenizer_dir = Path(tokenizer_dir)
            else:
                self.tokenizer_dir = tokenizer_dir

            if self.model_type is None:
                try:
                    self.model_type = self.tokenizer_dir.parent.name
                except:
                    self.model_type = None

            if PLOTS and self.model_type is None:
                raise ValueError("-p flag requires to correctly set the model type")

            # The node_config for the starter is the whole json! It should know the
            # other nodes in the network to initialize them
            self.role = "starter"
            self.own_config = node_config["nodes"]["starter"]

            self._select_device(model_device)

            self.n_nodes = 1 + (
                0
                if "secondary" not in node_config["nodes"]
                else len(node_config["nodes"]["secondary"])
            )
            self.next_node = (
                None if self.n_nodes == 1 else node_config["nodes"]["secondary"][0]
            )
            self.prev_node = (
                None if self.n_nodes == 1 else node_config["nodes"]["secondary"][-1]
            )

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
            self._init_model(
                self.n_layers_local,
                model_path=self.model_path,
            )

            # Initialize tokenizer
            self._load_tokenizer(self.tokenizer_dir)

            # Standalone:
            if self.n_nodes == 1:
                self.out_message_queue = self.in_message_queue
                self.out_queue_not_empty = self.in_queue_not_empty
        else:
            # model_config and chunk_path may be absent!
            self.model_config = model_config  # May be None
            if isinstance(chunk_path, str):
                self.model_path = Path(chunk_path)
            else:
                self.model_path = chunk_path  # May be None

            # Parse role name to get right node config
            split_node_type = node_type.split(":")
            if len(split_node_type) == 1:
                if len(node_config["nodes"]["secondary"]) > 1:
                    raise ValueError(
                        "Need to specify which of the secondary nodes this is"
                    )
                elif (
                    "secondary" in node_config["nodes"]
                    and len(node_config["nodes"]["secondary"]) == 1
                ):
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
                if "nodes" not in node_config
                else node_config["nodes"]["secondary"][secondary_index]
            )
            self.starter_addr = self.own_config["communication"]["starter_addr"]
            self._select_device(model_device)
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
        cp.engine.exit()

    # ---------------------------------------------------------------------------------

    def launch_starter(
        self, n_samples: int, max_tokens: int, prompt: Optional[str] = None
    ) -> Tuple[List[str], List[Tuple[int, float]]]:
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
        self.n_samples = n_samples
        self.inference_thread = threading.Thread(
            target=self.start_inference,
            args=(n_samples,),
            kwargs={
                "max_new_tokens": max_tokens,
                "prompt": prompt,
                "metrics": metrics_dict,
            },
        )
        # NOTE: the separate thread is just a placeholder to make the interface uniform
        # for all nodes - here we wait for the processing loop to conclude!
        self.inference_thread.start()
        self.inference_thread.join()
        self.shutdown()
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
        assert self.model_config is not None and self.model is not None

        if self.conn_to_next:
            self.conn_to_next.shutdown()
            self.conn_to_next = None
        if self.conn_to_prev:
            self.conn_to_prev.shutdown()
            self.conn_to_prev = None

        # Configuration for all nodes
        self._create_sockets()

        # Differentiate between different types
        if self.node_type == "starter":
            assert max_new_tokens is not None

            self.n_samples = n_samples
            self.running.set()
            self._launch_queue_threads()

            if VERB:
                print(
                    f"[INFO] Starting generation loop - {n_samples} samples, {max_new_tokens} tokens each"
                )
            logger_wp.info("Starting generation loop")

            out_text, gen_time = self._starter_loop(
                n_samples, prompt, max_new_tokens=max_new_tokens
            )

            if metrics is not None:
                # NOTE: this allows to return values even if this method is on a
                # separate thread! Just read from this object after `join`
                metrics["gen_text"] = out_text
                metrics["gen_time"] = gen_time
        else:
            assert self.next_node is not None and self.prev_node is not None
            # Secondary node
            self.running.set()
            self._launch_queue_threads()
            if VERB:
                print("[INFO] Starting generation loop")
            logger_wp.info("Starting generation loop")
            self._secondary_loop()

    def stop_generation(self) -> int:
        try:
            time.sleep(2)
            self.running.clear()  # Redundant, but ok
            if "starter" not in self.role:
                if VERB:
                    print("Stopping main thread")
                self.inference_thread.join()
            if self.n_nodes > 1 and self.conn_to_prev and self.conn_to_next:
                if VERB:
                    print("Stopping input queue thread")
                self.conn_to_prev.shutdown()
                if VERB:
                    print("Stopping output queue thread")
                self.conn_to_next.shutdown()
            return 1
        except:
            return 0

    def shutdown(self) -> int:
        """
        Turn off the node - stop server, close sockets and stop thread.

        Returns:
            1 upon success, 0 otherwise (exception gets raised)
        """
        if VERB:
            print("[INFO] Shutting down")

        try:
            assert self.stop_generation()
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

        Note: for standalone (single node) operation, the connections are not created,
        and therefore no threads are launched.
        """
        start_only = self.node_type == "starter" and (
            not self.conn_to_prev and not self.conn_to_next
        )
        assert start_only == (
            self.n_nodes == 1
        ), "Not running in standalone mode, but missing connections"

        if not start_only:
            assert self.conn_to_next and self.conn_to_prev
            if VERB:
                print("[INFO] Starting queue threads")

            self.conn_to_prev.launch()
            self.conn_to_next.launch()

    def _create_sockets(self):
        """
        Create sockets for communicating the intermediate results with the previous and
        next nodes in the chain.

        Starter nodes will open the connection towards the next node first, while all
        other nodes will first connect to the previous ones (otherwise the application
        would just wait indefinitely, as no node will connect with any other).
        """
        assert self.conn_to_prev is None and self.conn_to_next is None

        if self.node_type != "starter" and (not self.prev_node or not self.next_node):
            raise RuntimeError("Missing neighboring node info!")

        if self.node_type == "starter":
            # Only create socket if NOT in standalone mode
            if self.next_node is not None and self.n_nodes != 1:
                self.conn_to_next = OutputNodeConnection(
                    self.own_config,
                    next_node=self.next_node,
                    queue=self.out_message_queue,
                    event_callback=self.out_queue_not_empty,
                    verb=VERB,
                )
        else:
            assert self.next_node is not None and self.prev_node is not None

        if self.prev_node is not None:
            self.conn_to_prev = InputNodeConnection(
                self.own_config,
                prev_node=self.prev_node,
                queue=self.in_message_queue,
                event_callback=self.in_queue_not_empty,
                verb=VERB,
            )

        if self.node_type != "starter":
            self.conn_to_next = OutputNodeConnection(
                self.own_config,
                next_node=self.next_node,
                queue=self.out_message_queue,
                event_callback=self.out_queue_not_empty,
                verb=VERB,
            )

    def _build_msg(self, data: Any, sample_index: int, stop: bool = False) -> Dict:
        """
        Build the message which is transmitted to the next node.

        Args:
            data: the activations to be transmitted
            sample_index: index of the current sample (allows to check)

        Returns:
            the message - a Python dict with the fields "sample_index" and
            "data"
        """
        return {"sample_index": sample_index, "data": data, "stop": stop}

    # ----- Private -------------------------------------------------------------------

    def _select_device(self, device):
        """
        Select the torch device to be used to load and process the model.
        Priority (high to low):
            1. Command line arg (`--device`)
            2. Config file
            3. Default device
        """
        # Possibly get device info if found in config file
        try:
            self.model_device = device if device else self.own_config["device"]
        except KeyError:
            warnings.warn(f"Using default device {DEFAULT_DEVICE}")
            self.model_device = DEFAULT_DEVICE
        self.torch_model_device = torch.device(self.model_device)
        if VERB:
            print(f"Using device: {self.model_device}")

    def _init_model(
        self,
        n_transf_layers: int,
        *,
        model_path: Optional[Path] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the node's model chunk and move it to the target device
        (self.model_device).

        Args:
            model_parameters: state dict containing the weights - will be emptied
            n_transf_layers: number of transformer layers of the local model; required
                for initializing the submodel.
            *
            model_path: if present, it is the path in the local file system where
                the model chunk is found
            model_parameters: alternatively, the model parameters may already be present in
                memory (e.g., loaded previously or received by starter node)
        """
        assert self.model_config is not None, "No model configuration was found!"
        assert self.model is None, "The model was already initialized!"
        assert self.model_device is not None, "No device was specified"

        if not (model_path or model_parameters):
            raise ValueError(
                "At least one between model_path and model_parameters must be nonempty"
            )

        # TODO:
        # 1. load empty model
        # 2. load weights from disk
        # 3. if the dtype was overridden, cast weights
        # 4. load parameters to model
        if VERB:
            print("Initializing empty local model")

        Model_class = StarterNode if "starter" in self.node_type else SecondaryNode
        # self.model = Model_class(self.model_config, n_transf_layers).to_empty(device=self.model_device)
        with torch.device("meta"):
            self.model = Model_class(self.model_config, n_transf_layers)

        if VERB:
            print("Loading parameters")

        if model_path:
            # By default, use cpu
            model_parameters = load_sd(model_path)
        assert model_parameters is not None

        # Check n. of detected parameters are consistent with chunk (count_transformer_blocks)
        n_layers_detect = count_transformer_blocks(model_parameters)
        if n_transf_layers != n_layers_detect:
            raise ValueError(
                f"The number of detected transformer blocks ({n_layers_detect}) is "
                f"different from the expected one {n_transf_layers}; please "
                "re-run the model partition or check the configuration!"
            )

        if self.use_default_dtype:
            self.ptdtype = torch.float32
            if all([v.dtype == torch.float16 for v in model_parameters.values()]):
                self.ptdtype = torch.float16
            elif all([v.dtype == torch.bfloat16 for v in model_parameters.values()]):
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    self.ptdtype = torch.bfloat16

        model_dtype = self.ptdtype

        model_parameters = {k: v.to(model_dtype) for k, v in model_parameters.items()}

        if VERB:
            print(f"Using dtype {model_dtype}")
            print("Loading weights")

        self.model.load_weights(model_parameters, assign=True)
        if self.max_seq_length:
            print(f"[DEBUG] Truncating context length to {self.max_seq_length}")
            self.model.max_seq_length = self.max_seq_length
        else:
            # Use default value
            self.max_seq_length = self.model.max_seq_length

        if VERB:
            print(f"Moving model to {self.torch_model_device}")
        self.model = self.model.to(model_dtype).to(self.torch_model_device)
        print(self.model.state_dict())
        # self.model = self.model.to_empty(device=self.torch_model_device)
        # print(self.model.state_dict())

        if self.compile and hasattr(torch, "compile"):
            if VERB:
                print("Compiling local model - this may take a while", end="\r")
            try:
                self.model = torch.compile(self.model)
                if VERB:
                    print("Model compiled!                                ")
            except RuntimeError as e:
                warnings.warn(f"Unable to compile model! {e}")
        elif self.compile and not hasattr(torch, "compile"):
            from importlib.metadata import version

            warnings.warn(
                f"Installed torch version ({version('torch')}) does not support compiling models"
            )

        del model_parameters
        model_parameters = None
        gc.collect()

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
        # FIXME: this is just to give priority to HF; some tokenizer_config.json files (Llama 2) are broken...
        try:
            self.tok = Tokenizer(tokenizer_dir, force_backend="huggingface")
        except:
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
            print(f"Prompt style: {type(self.prompt_style)}")

        self.stop_tokens = self.prompt_style.stop_tokens(self.tok)
        if VERB:
            print("Tokenizer and prompt style have been loaded!")

    def _init_sample_caches(self, id, idx):
        """
        Initialize the model cache for the new sample `idx` with ID: `id`, using a
        specified dtype.

        Args:
            id: sample ID
            idx: new sample (encoded prompt)
            dtype: desired dtype for the KV caches

        Returns:
            Cache length (T_i)
            Input position tensor (input_pos)
            KV cache for the sumbodel (kvcaches)
        """
        assert self.model is not None

        self.T_i[id] = idx.size(1)
        self.input_pos[id] = torch.arange(
            0, self.T_i[id], device=self.torch_model_device
        )
        kvc_sublist: List[KVCache] = []
        for _, block in enumerate(self.model.transformer.h):
            # Build kv cache individually for each attn layer
            kvc_sublist.append(
                block.attn.build_kv_cache(
                    batch_size=1,
                    max_seq_length=self.model.max_seq_length,
                    rope_cache_length=self.model.cos.size(-1),
                    device=self.torch_model_device,
                    dtype=self.ptdtype,
                )
            )
        self.kvcaches[id] = kvc_sublist

    # ---- Main Loops -----------------------------------------------------------------

    def _starter_loop(
        self, n_samples: int, prompt: Optional[str] = None, **kwargs
    ) -> Tuple[List[str], List[Tuple[int, float]]]:
        """
        Generation loop for the starter node only.

        Args:
            n_samples: number of produced samples
            prompt: either the prompt itself or a string of the type "FILE:<prompt.txt>"
                containing each prompt as a separate paragraph

        Returns:
            list containing the `n_nodes` generated samples
            total generation time in seconds
        """
        assert self.model_config is not None and self.model is not None
        assert self.model_device is not None

        #
        # TODO
        # Starter loop should become agnostic of n_samples (it will work on-demand
        #
        # The prompt will need to be processed "outside", e.g., by the POST
        #
        # Will probably be able to "unify" this part of code before the actual loop for
        # both types of nodes - then _starter_loop and _secondary_loop could be
        # streamlined to just contain what's inside the "while"
        #
        if n_samples < 1:
            raise ValueError("Cannot generate less than 1 sample!")
        elif n_samples < self.n_nodes:
            warnings.warn(
                f"Generating less samples ({n_samples}) than nodes ({self.n_nodes}) will not be efficient!"
            )
        self.n_samples = n_samples

        if "cuda" in self.model_device:
            device_type = "cuda"
        elif "mps" in self.model_device:
            device_type = "mps"
        else:
            device_type = "cpu"
        ctx = (  # Use autocast if on cuda or cpu (MPS not supported yet)
            nullcontext()
            if device_type == "mps"
            else torch.amp.autocast(device_type=device_type, dtype=self.ptdtype)
        )

        # <<<< TODO: replace - analogous operations to be executed in POST from user
        # The POST should
        # Encode starting sequence - with prompt support
        if prompt is None:
            start = ["\n"] * n_samples
            start_styled = [self.prompt_style.apply(s) for s in start]
        else:
            start_styled = get_user_prompt(
                prompt, n_samples, prompt_style=self.prompt_style
            )

        assert len(start_styled) == n_samples

        idx = [
            self.tok.encode(txt, device=self.torch_model_device).view(1, -1)
            for txt in start_styled
        ]
        # >>>>

        # Initialize RoPE cache and attention mask
        self.model.init_rope_mask(device=self.torch_model_device)
        self.model.eval()

        # Starter Only
        self.samples: Dict[int, torch.Tensor] = {}
        self.prompt_lengths: Dict[int, int] = {}

        # >>>> TODO: remove - will be done in POST
        for i, samp in enumerate(idx):
            self.in_message_queue.append(self._build_msg(samp, i))
            self.prompt_lengths[i] = len(samp.squeeze())  # Length in tokens
            self.iter_ind[i] = 0
        self.in_queue_not_empty.set()

        if "max_new_tokens" in kwargs:
            # NOTE: can override the max. n. of tokens - must ensure
            self.max_new_tokens = {
                i: kwargs["max_new_tokens"] for i in range(len(self.prompt_lengths))
            }
            # Check max_new_tokens won't cause errors later
            if not all(
                [
                    self.max_new_tokens[i] + self.prompt_lengths[i]
                    <= self.model.max_seq_length
                    for i in range(n_samples)
                ]
            ):
                raise ValueError(
                    f"Cannot generate {kwargs['max_new_tokens']} tokens - would exceed block size!"
                )
        else:
            # The maximum number of tokens is the model's sequence length - prompt length
            self.max_new_tokens = {
                i: (self.model.max_seq_length - p_l)
                for i, p_l in self.prompt_lengths.items()
            }
            assert all(
                max_tok > 0 for max_tok in self.max_new_tokens.values()
            ), "Some prompt is longer than the context length of the model"
        # <<<<

        event_stop = threading.Event()
        loading_thread = threading.Thread(
            target=waiting_animation, args=("Processing samples", event_stop)
        )

        start_time = time.time()
        n_tokens = 0
        if PLOTS:
            self.tok_time.append((0, 0))

        print("[INFO] Launching processing loop")
        loading_thread.start()
        with torch.inference_mode(), ctx, catch_loop_errors(
            running_event=self.running, event_to_be_set=[event_stop]
        ):
            while self.running.is_set():
                # Wait for queue to contain msg -- timeout allows to handle shutdown
                if self.in_queue_not_empty.wait(timeout=2):
                    in_msg = self.in_message_queue.popleft()
                    if len(self.in_message_queue) < 1:
                        self.in_queue_not_empty.clear()

                    if in_msg["stop"]:
                        # The stopping message made the whole loop
                        self.running.clear()  # TODO: remove
                    else:
                        sample_id = in_msg["sample_index"]
                        idx = in_msg["data"].to(self.model_device)
                        stopping_detected = False

                        # Keep variable iter_ind[i] for each sample i
                        if self.iter_ind[sample_id] >= 1:
                            print("OUTPUT")
                            # We are not in the first iteration for this sample
                            # --> Can start processing messages from last secondary node
                            logits = self.model(idx, first_pass=False)
                            idx_next = sample(
                                logits,
                                temperature=self.temperature,
                                top_k=self.top_k,
                            )
                            idx_next = idx_next.view(1, -1)
                            self.samples[sample_id] = torch.cat(
                                (self.samples[sample_id], idx_next), dim=1
                            )
                            # Detect stopping token sequence and possibly interrupt gen for current sample
                            stopping_detected = detect_stop_tokens(
                                self.samples[sample_id], self.stop_tokens
                            )
                            # Update input pos (will be used in next pass)
                            self.input_pos[sample_id] = self.input_pos[sample_id][
                                -1:
                            ].add_(1)

                            # Only add new token after it has been generated
                            n_tokens += 1
                            if PLOTS:
                                self.tok_time.append(
                                    (n_tokens, time.time() - start_time)
                                )

                        else:
                            # First iteration for the current sample!
                            # Begin list of samples
                            self.samples[sample_id] = idx.view(1, -1)
                            # First iter for this sample, init KV cache!
                            self._init_sample_caches(sample_id, self.samples[sample_id])

                        # Send to next iff not at the last token
                        if self.iter_ind[sample_id] < self.max_new_tokens[sample_id]:
                            # Only propagate last token (KV cache) - OR all initial prompt if
                            # 1st iter
                            idx_cond = (
                                self.samples[sample_id]
                                if self.iter_ind[sample_id] == 0
                                else self.samples[sample_id][:, -1].view(1, -1)
                            )

                            # NOTE: Swap KVCache for correct sample
                            curr_kvcache = self.kvcaches[sample_id]
                            for ind_b, block in enumerate(self.model.transformer.h):
                                block.attn.kv_cache = curr_kvcache[ind_b]

                            # Forward in local model (first piece)
                            idx_cond = self.model(idx_cond, self.input_pos[sample_id])

                            # Send message
                            out_msg = self._build_msg(idx_cond, sample_id)
                        else:
                            # Generation finished
                            print(f"[DEBUG] Finished sample {sample_id}")
                            # TODO: decode and place msg in self.resp
                            # Also set resp_queue_not_empty to advertise sample

                            # Transmit msg with 'stop': true for this sample ID
                            out_msg = self._build_msg(
                                data="", sample_index=sample_id, stop=True
                            )

                        # UPDATE ITERATION COUNT FOR SAMPLE
                        self.iter_ind[sample_id] += 1

                        # NOTE: message queues will be the same if running in standalone!
                        self.out_message_queue.append(out_msg)
                        self.out_queue_not_empty.set()

        if VERB:
            print("[INFO] Generation completed!                          ")
        logger_wp.info("Generation completed")

        out_truncated = [
            find_eot(smp, self.stop_tokens, self.prompt_lengths[i])
            for i, smp in self.samples.items()
        ]
        if VERB:
            print("Truncated samples:")
            for i, smp in enumerate(out_truncated):
                print(
                    f"- Sample {i} truncated to {len(smp.squeeze())}/{len(self.samples[i].squeeze())}"
                )
        out_samples = [self.tok.decode(smp) for smp in out_truncated]

        return out_samples, self.tok_time

    def _secondary_loop(self):
        """
        Execution loop for non-starter nodes. This method must be used as the target of
        a thread that is launched once the node has been correctly initialized.

        The execution will be stopped once a PUT request is made to /stop.
        """
        assert self.conn_to_prev is not None and self.conn_to_next is not None
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
        ctx = (  # Use autocast if on cuda or cpu (MPS not supported yet)
            nullcontext()
            if device_type == "mps"
            else torch.amp.autocast(device_type=device_type, dtype=self.ptdtype)
        )

        # Allow node to be 100% agnostic of the system! If it receives a sample with a
        # new ID, it will initialize the caches for that sample on the fly!
        self.model.init_rope_mask(device=self.torch_model_device)
        self.model.eval()

        event_stop = threading.Event()
        loading_thread = threading.Thread(
            target=waiting_animation, args=("Processing samples", event_stop)
        )
        iter = 0
        first_glob_iter = True  # True for the first n_samples iters

        print("[INFO] Launching processing loop")
        loading_thread.start()
        with ctx, torch.inference_mode(), catch_loop_errors(
            running_event=self.running, event_to_be_set=[event_stop]
        ):
            while self.running.is_set():
                if self.in_queue_not_empty.wait(timeout=2):
                    # Extract message from queue
                    in_msg = self.in_message_queue.popleft()
                    if len(self.in_message_queue) <= 0:
                        self.in_queue_not_empty.clear()

                    sample_id = in_msg["sample_index"]

                    if "stop" in in_msg and in_msg["stop"]:
                        print(f"[DEBUG] Finished sample {sample_id}")
                        # TODO: delete variables for sample id

                        self.out_message_queue.append(in_msg)
                        self.out_queue_not_empty.set()
                    else:
                        if iter >= self.n_samples:
                            first_glob_iter = False

                        idx = in_msg["data"].to(self.torch_model_device)
                        if sample_id not in self.T_i:
                            assert (
                                first_glob_iter
                            ), "Should have seen this sample already..."
                            # Initialization of the input_pos
                            self._init_sample_caches(sample_id, idx)

                        # Swap KVCache
                        curr_kvcache = self.kvcaches[sample_id]
                        for ind_b, block in enumerate(self.model.transformer.h):
                            block.attn.kv_cache = curr_kvcache[ind_b]

                        # Forward pass
                        outs = self.model(idx, input_pos=self.input_pos[sample_id])

                        # Build msg
                        out_msg = self._build_msg(outs, sample_id)
                        # Send to next
                        self.out_message_queue.append(out_msg)
                        self.out_queue_not_empty.set()

                        self.input_pos[sample_id] = self.input_pos[sample_id][-1:].add_(
                            1
                        )
                        iter += 1

        if VERB:
            print("Node inference loop stopped")

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
        - Secondary nodes:
            Receive configuration info from the starter node and start connection with
            previous and next, then start generation, i.e., wait for incoming data
            through the sockets to be passed through the local model chunk.
            Message fields:
                role ("secondary:n" - does NOT overwrite the previous node given at init)
                prev_node (as in configuration json file)
                next_node (as in configuration json file)
                model_config (serialized with Config.asdict())
                n_nodes (number of total network nodes)
                n_local_layers (n. of chunk layers - prevent issues due to different config)
                [params] (model parameters - needed if no chunk path was passed)
                n_samples (number of produced samples)
        """
        if (
            self.node_type is None or "secondary" in self.node_type
        ) and self.model is None:  # Only for non-init nodes
            if len(path) > 0 and path[0] == "init":
                assert not self.running.is_set()
                init_msg = pickle.loads(cp.request.body.read())
                if self.node_type is None:
                    self.role = self.node_type = init_msg["role"]
                self.prev_node = init_msg["prev_node"]
                self.next_node = init_msg["next_node"]
                # Assume model config is not initialized
                self.model_config = Config(**init_msg["model_config"])
                self.n_nodes = init_msg["n_nodes"]
                self.n_layers_local = init_msg["n_local_layers"]
                self.max_seq_length = (
                    None
                    if "max_seq_length" not in init_msg
                    else init_msg["max_seq_length"]
                )

                if "params" in init_msg:
                    if VERB:
                        print("Received parameters from starter")
                    self._init_model(
                        self.n_layers_local, model_parameters=init_msg["params"]
                    )
                    # Clear memory of model_params
                    del init_msg["params"]
                    init_msg["params"] = None
                    gc.collect()
                else:
                    if self.model_path is None:
                        raise RuntimeError(
                            "The received message did not contain the model parameters "
                            "- please specify a model chunk path when initializing "
                            "GPTServer object"
                        )
                    if VERB:
                        print("Loading parameters from disk")
                    self._init_model(self.n_layers_local, model_path=self.model_path)

                self.n_samples = init_msg["n_samples"]

                if VERB:
                    print(f"{self.n_nodes} Nodes, generating {self.n_samples} samples")

                if VERB:
                    print(f"[INFO] Starting operation - {self.node_type} node")
                # FIXME: review threads
                logger_wp.info("Received initialization information!")
                self.inference_thread = threading.Thread(
                    target=self.start_inference, daemon=True, args=(self.n_samples,)
                )
                self.inference_thread.start()
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
        if self.node_type == "starter":
            raise cp.HTTPError(501, "PUT not implemented!")
        else:
            # TODO: fix shutdown procedure - should also release models if not terminating app
            if len(path) > 0 and path[0] == "stop":
                self._end_thr = threading.Thread(target=self.shutdown)
                self._end_thr.start()
                # self._end_thr.join()  # cannot wait, since thread stops server
                if VERB:
                    print("[INFO] Node stopped through PUT request!")
                logger_wp.info("Received stopping directive")
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found!")

    def DELETE(self):
        """Not implemented"""
        raise cp.HTTPError(501, "DELETE not implemented!")
