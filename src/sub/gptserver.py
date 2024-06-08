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
from sub.config import DTYPE, N_LAYERS_NODES, TEMPERATURE, TOP_K
from sub.connections import InputNodeConnection, OutputNodeConnection
from sub.model import Config, KVCache, sample
from sub.prompts import (PromptStyle, get_user_prompt, has_prompt_style,
                         load_prompt_style)
from sub.submodels import SecondaryNode, StarterNode
from sub.tokenizer import Tokenizer
from sub.typing import FileType
from sub.utils import (count_transformer_blocks, find_eot, load_sd,
                       loading_bar, plot_tokens_per_time)

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
    n_samples: Optional[int] = None
    model_type = None

    # True iff the model has been initialized and it is ready to perform inference.
    running: bool = False

    # Connections
    conn_to_next: Optional[OutputNodeConnection] = None
    conn_to_prev: Optional[InputNodeConnection] = None

    msg_format = {"sample_index": 0, "data": None, "stop": False}

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

        self.compile = False if "compile" not in kwargs else kwargs["compile"]

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
            self._load_tokenizer(self.tokenizer_dir)

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
            assert self.next_node is not None and self.prev_node is not None
            # Secondary node
            self.running = True
            self._launch_queue_threads()
            if VERB:
                print("[INFO] Starting generation loop")
            logger_wp.info("Starting generation loop")
            self._secondary_loop()

    def stop_generation(self) -> int:
        try:
            time.sleep(2)
            self.running = False  # Redundant, but ok
            if "starter" not in self.role:
                if VERB:
                    print("Stopping main thread")
                self.inference_thread.join()
            if self.n_nodes > 1:
                if VERB:
                    print("Stopping input queue thread")
                self.conn_to_prev.shutdown()
                if VERB:
                    print("Stopping input queue thread")
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

        model_dtype = torch.float32
        if all([v.dtype == torch.float16 for v in model_parameters.values()]):
            model_dtype = torch.float16
        elif all([v.dtype == torch.bfloat16 for v in model_parameters.values()]):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16

        if VERB:
            print("Initializing local model")
            print(f"Using {model_dtype}")

        Model_class = StarterNode if "starter" in self.node_type else SecondaryNode
        self.model = Model_class(self.model_config, n_transf_layers)
        if model_dtype in {torch.float16, torch.bfloat16}:
            self.model = self.model.to(model_dtype)
        self.model.load_weights(model_parameters)
        self.model = self.model.to(self.torch_model_device)

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
        prompt_lengths = {i: len(id.squeeze()) for i, id in enumerate(idx)}

        # Initialize RoPE cache and attention mask
        self.model.init_rope_mask(device=self.torch_model_device)
        # Prompt size T and input_pos are now lists of n_samples elements
        T_i: Mapping[int, int] = {}  # Contains size of context of each prompt
        input_pos: Mapping[int, torch.Tensor] = (
            {}
        )  # Will contain the input pos. of all samples
        kvcaches: Mapping[int, List[KVCache]] = {}
        self.model.eval()

        start_time = time.time()
        if PLOTS:
            self.tok_time.append((0, 0))
        with torch.no_grad():
            with ctx:
                total_iters = max_new_tokens * n_samples
                first_glob_iter = True
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
                        first_glob_iter = False
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
                        idx_next = idx_next.view(1, -1)
                        idx[sample_id] = torch.cat((idx[sample_id], idx_next), dim=1)
                        input_pos[sample_id] = input_pos[sample_id][-1:].add_(1)
                    else:
                        # First iter for this sample, init KV cache!
                        T_i[k] = idx[sample_id].size(1)
                        input_pos[k] = torch.arange(
                            0, T_i[sample_id], device=self.torch_model_device
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
                                    dtype=ptdtype,
                                )
                            )
                        kvcaches[k] = kvc_sublist

                    # Send to next iff not at the last token
                    if k < (n_samples * (max_new_tokens - 1)):
                        # NOTE: no support for ctx > block_size
                        # idx_cond should be equal to idx_next if after 1st global iter
                        idx_cond = (
                            idx[sample_id]
                            if first_glob_iter
                            else idx[sample_id][:, -1].view(1, -1)
                        )

                        # NOTE: Swap KVCache for correct sample
                        curr_kvcache = kvcaches[sample_id]
                        for ind_b, block in enumerate(self.model.transformer.h):
                            block.attn.kv_cache = curr_kvcache[ind_b]

                        # Forward in local model (first piece)
                        idx_cond = self.model(idx_cond, input_pos[sample_id])

                        # Send message
                        out_msg = self._build_msg(idx_cond, sample_id)
                        if self.conn_to_next and self.n_nodes > 1:
                            self.out_message_queue.append(out_msg)
                            self.out_queue_not_empty.set()
                        else:
                            # Single-node scenario
                            self.in_message_queue.append(out_msg)
                            self.in_queue_not_empty.set()

        tot_time = time.time() - start_time
        if PLOTS:
            self.tok_time.append((total_iters, tot_time))
            # Store plotted points as csv file
            os.makedirs(os.path.join(script_dir, "..", "logs"), exist_ok=True)
            points_file_path = os.path.join(
                script_dir,
                "..",
                "logs",
                f"tokens_time_samples_{self.n_nodes}nodes_{self.model_type}_{n_samples}samples.csv",
            )
            if not os.path.exists(os.path.dirname(points_file_path)):
                os.makedirs(os.path.dirname(points_file_path), exist_ok=True)
            with open(points_file_path, "w") as f:
                times = [x[1] for x in self.tok_time]
                n_tok = [x[0] for x in self.tok_time]
                for i in range(len(times)):
                    f.write(f"{times[i]},{n_tok[i]}\n")

            plot_tokens_per_time(
                self.tok_time,
                out_path=os.path.join(
                    script_dir,
                    "..",
                    "img",
                    f"tokens_time_{self.n_nodes}nodes_{self.model_type}_{n_samples}samples.png",
                ),
            )

        # Send stop message to the next (no queue used)
        if VERB:
            print("[INFO] Sending stopping message over socket  ")
        self.out_message_queue.append(self._build_msg("", -1, stop=True))
        self.out_queue_not_empty.set()
        self.running = False
        if VERB:
            print("[INFO] Generation completed!                          ")
        logger_wp.info("Generation completed")

        out_truncated = [
            find_eot(smp, self.stop_tokens, prompt_lengths[i])
            for i, smp in enumerate(idx)
        ]
        if VERB:
            print("Truncated samples:")
            for i, smp in enumerate(out_truncated):
                print(
                    f"- Sample {i} truncated to {len(smp.squeeze())}/{len(idx[i].squeeze())}"
                )
        out_samples = [self.tok.decode(smp) for smp in out_truncated]

        return out_samples, tot_time

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

        # Allow node to be 100% agnostic of the system! If it receives a sample with a
        # new ID, it will initialize the caches for that sample on the fly!
        self.model.init_rope_mask(device=self.torch_model_device)
        kvcaches: Mapping[int, List[KVCache]] = {}
        T_i: Mapping[int, int] = {}  # Contains size of context of each prompt
        input_pos: Mapping[int, torch.Tensor] = {}

        self.model.eval()

        loopsigns = ["|", "/", "-", "\\"]
        iter = 0
        first_glob_iter = True  # True for the first n_samples iters
        with torch.no_grad():
            with ctx:
                while self.running:
                    logger_wp.info(f"Iter {iter}")

                    assert self.in_queue_not_empty.wait()

                    # Extract message from queue
                    in_msg = self.in_message_queue.popleft()
                    if len(self.in_message_queue) <= 0:
                        self.in_queue_not_empty.clear()

                    if "stop" in in_msg and in_msg["stop"]:
                        print("[DEBUG] Received stopping message over socket")
                        self.running = False  # Redundant

                    if self.running:
                        sample_id = in_msg["sample_index"]
                        if iter >= self.n_samples:
                            first_glob_iter = False

                        idx = in_msg["data"].to(self.torch_model_device)
                        if sample_id not in T_i:
                            assert (
                                first_glob_iter
                            ), "Should have seen this sample already..."
                            # Initialization of the input_pos
                            T_i[sample_id] = idx.size(1)
                            input_pos[sample_id] = torch.arange(
                                0, T_i[sample_id], device=self.torch_model_device
                            )
                            kvc_sublist: List[KVCache] = []
                            for block in self.model.transformer.h:
                                # Build kv cache individually for each attn layer
                                kvc_sublist.append(
                                    block.attn.build_kv_cache(
                                        batch_size=1,
                                        max_seq_length=self.model.max_seq_length,
                                        rope_cache_length=self.model.cos.size(-1),
                                        device=self.torch_model_device,
                                        dtype=ptdtype,
                                    )
                                )
                            kvcaches[sample_id] = kvc_sublist

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
                        self.out_message_queue.append(
                            self._build_msg("", -1, stop=True)
                        )
                        self.out_queue_not_empty.set()
                        self.running = False

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
                assert not self.running
                init_msg = pickle.loads(cp.request.body.read())
                if self.node_type is None:
                    self.node_type = init_msg["role"]
                self.prev_node = init_msg["prev_node"]
                self.next_node = init_msg["next_node"]
                # Assume model config is not initialized
                self.model_config = Config(**init_msg["model_config"])
                self.n_nodes = init_msg["n_nodes"]
                self.n_layers_local = init_msg["n_local_layers"]

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
            if len(path) > 0 and path[0] == "stop":
                self._end_thr = threading.Thread(target=self.shutdown)
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
