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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cherrypy as cp
import torch

from sub.config import DEVICE as DEFAULT_DEVICE
from sub.config import (DTYPE, DTYPE_TORCH_MAPPING, N_LAYERS_NODES,
                        TEMPERATURE, TOP_K)
from sub.connections import InputNodeConnection, OutputNodeConnection
from sub.gpt_distr import GPTDistributed
from sub.model import Config, KVCache, sample
from sub.prompts import PromptStyle, has_prompt_style, load_prompt_style
from sub.submodels import SecondaryNode, StarterNode
from sub.tokenizer import Tokenizer
from sub.utils import (catch_loop_errors, count_transformer_blocks,
                       detect_stop_tokens, find_eot, get_available_models,
                       load_sd, s_to_ns, waiting_animation)
from sub.utils.typing import FileType, JSONObject, JSONType

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

    # Node info - returned in GET
    node_capabilities: JSONType = dict(node_config={}, role="", model={})

    # Connections - None if not connected -- not passed to GPTDistr
    conn_to_next: Optional[OutputNodeConnection] = None
    conn_to_prev: Optional[InputNodeConnection] = None

    # TODO: pass queues to GPTDistr
    # Input message queue
    in_message_queue = deque([])
    in_queue_not_empty = threading.Event()  # Replaces busy waiting
    in_queue_not_empty.clear()
    # Output message queue
    out_message_queue = deque([])
    out_queue_not_empty = threading.Event()
    out_queue_not_empty.clear()

    # Keep track of nodes
    # Items example:
    # {
    #   id:
    #   addr:
    #   communication: {
    #       port
    #   }
    #   inference: {
    #       port_in
    #       port_out
    #   }
    #   connected: (bool) <-- !
    #   capabilities:
    #   TODO
    # }
    nodes_registry = []

    # Web server
    webserv_config = {
        "/": {
            "request.dispatch": cp.dispatch.MethodDispatcher(),
            "tools.sessions.on": True,
        }
    }

    def __init__(
        self,
        node_config: Dict[str, Any],
        node_type: str,  # In secondary nodes, uninitialized: "secondary"
        ckpt_dir: FileType = script_dir / ".." / "checkpoints",
        *,
        max_seq_length: Optional[int] = None,
        model_device: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize a GPT Server, used to deploy a GPT Distributed instance and manage
        REST APIs and inference connections (sockets).

        Args:
            node_config: JSON-serializable node config.
            node_type: "starter" or "secondary" - for secondary nodes, it will be
                updated at initialization (POST from starter) by adding ":<n>" where <n>
                is the node index.
            ckpt_dir: directory containing the checkpoints (models); it should contain
                "org/model_name" folders.
            *
            model_device: allows to override the device specified in node_config (if
                any); typically associated with command line arg.
            dtype: allows to override the data type used for the model; defaults to
                bfloat16 if available (from device), else float16.
            kwargs: allow to override "verb" (bool), model sequence length
                ("model_seq_length") and "compile".
        """
        # Init GPTDistributed
        self.gptdistr = GPTDistributed(
            node_config,
            node_type,
            ckpt_dir,
            max_seq_length=max_seq_length,
            model_device=model_device,
            dtype=dtype,
            **kwargs,
        )
        self.role = node_type

        # TODO: other things??? E.g., "map" attributes (callbacks, like queues)
        # Map queues and related events
        self.gptdistr.init_msg_queues(
            self.in_message_queue,
            self.out_message_queue,
            self.in_queue_not_empty,
            self.out_queue_not_empty,
        )

        # Map 'running' event
        self.running = self.gptdistr.running

        # Device (str)
        self.model_device = self.gptdistr.model_device

        # TODO: get node's capabilities
        self.node_capabilities = self._get_node_capabilities()
        # But need to use self.gptdistr.own_config and self.gptdistr.ckpt_dir

    # ---------------------------------------------------------------------------------
    def wait_for_nodes_registration(self):
        """
        Listen for incoming PUT requests containing nodes information and update
        self.nodes_registry (set connected = true)
        """
        assert self.role == "starter"
        # TODO
        pass

    def register_at_starter(self):
        """
        Send PUT request to starter node containing capabilities.
        """
        assert self.role != "starter"
        # TODO
        pass

    def start_webserv(self):
        """
        Launch the web server.
        """
        cp.tree.mount(self, "/", self.webserv_config)
        cp.config.update(
            {
                "server.socket_host": self.gptdistr.own_addr,
                "server.socket_port": self.gptdistr.own_comm_port,
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

    def launch_starter(self):
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
        if "starter" not in self.role:
            raise ValueError(f"Cannot run `launch_starter` for node type {self.role}")
        self.inference_thread = threading.Thread(target=self.start_inference)

        # NOTE: the separate thread is just a placeholder to make the interface uniform
        # for all nodes - here we wait for the processing loop to conclude!
        self.inference_thread.start()
        self.inference_thread.join()
        self.shutdown()

    def start_inference(self):
        """
        This method is meant to be ran as an independent thread.

        Perform normal operation (open sockets, wait for communication from previous
        node and forward activations to next one).
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

        if VERB:
            print("[INFO] Starting generation loop")
        logger_wp.info("Starting generation loop")
        self.running.set()

        # Differentiate between different types - FIXME: needed?
        if "starter" in self.node_type:
            self._launch_queue_threads()
            self.gptdistr.starter_loop()
        else:
            assert self.next_node is not None and self.prev_node is not None
            # Secondary node
            self._launch_queue_threads()
            self.gptdistr.secondary_loop()

    def stop_generation(self) -> bool:
        """
        Interrupt the current application run.

        This method will:
        1. Stop main loop (non-starters)
        2. Stop connections (and queue threads)
        3. Delete model

        NOTE: this method does NOT turn the node off, it just un-initializes it; it will
        leave the HTTP server UP, allowing to further initialization calls from any
        starter node.
        """
        try:
            self.running.clear()
            if "starter" not in self.role:
                if VERB:
                    print("Stopping main thread")
                self.inference_thread.join()
            if self.n_nodes > 1 and self.conn_to_prev and self.conn_to_next:
                if VERB:
                    print("Stopping input queue thread")
                self.conn_to_prev.shutdown()
                self.conn_to_prev = None
                if VERB:
                    print("Stopping output queue thread")
                self.conn_to_next.shutdown()
                self.conn_to_next = None

            # TODO: maybe delete some run-specific parameters (e.g., role)
            # Clear node model
            self.model = None
            gc.collect()
            return True
        except:
            return False

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
                print("[INFO] Stopping HTTP server")
            self.stop_webserv()
            if VERB:
                print("[INFO] Closing application")
            return 1
        except:
            return 0

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

    def _build_serv_resp(
        self,
        generated_text: str,
        in_msg: Dict[str, Any],
        sample_id: int,
        tot_duration: int,
        load_duration: int,
        prompt_length: int,
        t_prompt_eval: int,
        len_tokens_gen: int,
        t_decode: int,
    ) -> Dict[str, Any]:
        """
        Package the generated text as specified by the Ollama APIs.

        All times should be expressed in NANOSECONDS.

        Ref:
        {
          "model": "llama3",
          "created_at": "2023-08-04T19:22:45.499127Z",
          "response": "",
          "done": true,
          "context": [1, 2, 3],
          "total_duration": 10706818083,
          "load_duration": 6338219291,
          "prompt_eval_count": 26,
          "prompt_eval_duration": 130079000,
          "eval_count": 259,
          "eval_duration": 4232710000
        }
        """
        out_msg = {}
        out_msg["model"] = in_msg["model"]  # FIXME: use model type
        out_msg["created_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        out_msg["response"] = generated_text
        out_msg["done"] = True
        out_msg["context"] = [sample_id]
        out_msg["total_duration"] = tot_duration
        out_msg["load_duration"] = load_duration
        out_msg["prompt_eval_count"] = prompt_length
        out_msg["prompt_eval_duration"] = t_prompt_eval
        out_msg["eval_count"] = len_tokens_gen
        out_msg["eval_duration"] = t_decode

        return out_msg

    def _get_node_capabilities(self) -> JSONType:
        """
        Return the node capabilities to be sent as response of GET requests.
        The info is a JSON-serializable dict.

        The dict is stored as a class attribute, this method updates it.

        Fields:
            - node_config: from configuration JSON (passed at init)
                - addr: IP
                - communication:
                    - port
                - inference:
                    - port_in
                    - port_out
                - [device]
            - role: ("secondary" if not init, else: "secondary:n") else, if starter: "starter"        FIXME
            - model:
                - active: "" if none, else HF name of the model, as rx at init
                - available: list of available models, with the structure:
                    "available": [
                      {
                        "name": "model_name",  <-- from checkpoints dir
                        "hf_config": {
                          "org": organization name (e.g., mistralai),
                          "name": actual model name
                        },
                        "chunks": {
                          "<n>nodes": [...],   <-- List of all the chunks (list dir)
                          "<k>nodes": [...]
                        }
                      },
                      ...
                    ]
            TODO
        """
        self.node_capabilities["node_config"] = self.gptdistr.own_config
        self.node_capabilities["role"] = self.gptdistr.role
        self.node_capabilities["model"] = {
            "active": self.gptdistr.model_type,
            "available": get_available_models(self.gptdistr.ckpt_dir),
        }
        return self.node_capabilities

    # ----- REST API ------------------------------------------------------------------

    def GET(self, *path):
        """
        Functions
            Return node information (port numbers, [capabilities]?)
            Used for pinging "neighbor" nodes
        """
        if not len(path):
            return json.dumps(self._get_node_capabilities())
        else:
            raise cp.HTTPError(404, "Not found")

    def POST(self, *path):
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

        - Starter node: Ollama-like APIs
        """
        if (
            self.node_type is None or "secondary" in self.node_type
        ) and self.model is None:  # Only for non-init nodes
            if len(path) > 0 and path[0] == "init":
                assert not self.running.is_set()
                init_msg = pickle.loads(cp.request.body.read())
                if self.node_type is None:
                    self.node_type = init_msg["role"]
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

                if VERB:
                    print(f"[INFO] Starting operation - {self.node_type} node")

                self.inference_thread = threading.Thread(
                    target=self.start_inference, daemon=True
                )
                self.inference_thread.start()
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found")
        elif self.role == "starter":
            if not len(path):
                return "Available URLs: "
            if path[0] == "api":
                # Ollama-like APIs
                if path[1] == "generate":
                    content_type = cp.request.headers["Content-Type"]
                    raw_body = cp.request.body.read().decode("utf-8")
                    if "application/json" in content_type:
                        body = json.loads(raw_body)
                    else:
                        return "Unsupported Media Type", 415
                    body = json.loads(raw_body)
                    resp = self.process_user_prompt(body)
                    return json.dumps(resp)
            elif path[0] == "register":
                # TODO: Secondary node registering at starter node
                pass

            raise cp.HTTPError(404, "Not found")

        elif self.model is not None:  # FIXME
            raise cp.HTTPError(
                403,
                f"Failed to configure node - the model was already initialized: {self.node_type}",
            )
        else:
            raise cp.HTTPError(404, "Not found")

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
