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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cherrypy as cp
import requests
import torch

from sub.config import N_LAYERS_NODES
from sub.connections import InputNodeConnection, OutputNodeConnection
from sub.gpt_distr import GPTDistributed
from sub.model import Config
from sub.utils import (catch_loop_errors, count_transformer_blocks,
                       detect_stop_tokens, find_eot, get_available_models,
                       load_sd, s_to_ns, waiting_animation)
from sub.utils.typing import FileType, JSONObject, JSONType
from sub.utils.utils import (get_chunk_path,
                             is_model_chunk_available_secondary, load_from_pt,
                             load_model_config, split_and_store)

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
    node_capabilities: JSONType = dict(node_config={}, role="", model={}, last_update=0)

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

    n_samples = 0

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
        self.verb = VERB if "verb" not in kwargs else kwargs["verb"]

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
        self.node_config = node_config
        self.ckpt_dir = Path(ckpt_dir)

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
        self.own_address = self.node_config["addr"]
        self.own_comm_port = int(self.node_config["communication"]["port"])
        # But need to use self.gptdistr.own_config and self.gptdistr.ckpt_dir

        self.start_webserv()

    # ---------------------------------------------------------------------------------

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

    def start_inference(self):
        """
        This method is meant to be ran as an independent thread.

        Perform normal operation (open sockets, wait for communication from previous
        node and forward activations to next one).
        """
        assert self.gptdistr.running.is_set()

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
        if "starter" in self.role:
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
        start_only = self.role == "starter" and (
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

        if self.role != "starter" and (not self.prev_node or not self.next_node):
            raise RuntimeError("Missing neighboring node info!")

        if self.role == "starter":
            # Only create socket if NOT in standalone mode
            if self.next_node is not None and self.n_nodes != 1:
                self.conn_to_next = OutputNodeConnection(
                    self.node_config,
                    next_node=self.next_node,
                    queue=self.out_message_queue,
                    event_callback=self.out_queue_not_empty,
                    verb=VERB,
                )
        else:
            assert self.next_node is not None and self.prev_node is not None

        if self.prev_node is not None:
            self.conn_to_prev = InputNodeConnection(
                self.node_config,
                prev_node=self.prev_node,
                queue=self.in_message_queue,
                event_callback=self.in_queue_not_empty,
                verb=VERB,
            )

        if self.role != "starter":
            self.conn_to_next = OutputNodeConnection(
                self.node_config,
                next_node=self.next_node,
                queue=self.out_message_queue,
                event_callback=self.out_queue_not_empty,
                verb=VERB,
            )

    def _build_serv_resp(
        self, in_msg: Dict[str, Any], distr_response: Tuple[Any, ...]
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
        out_msg["response"] = distr_response[0]
        out_msg["done"] = True
        out_msg["context"] = [distr_response[1]]
        out_msg["total_duration"] = distr_response[2]
        out_msg["load_duration"] = distr_response[3]
        out_msg["prompt_eval_count"] = distr_response[4]
        out_msg["prompt_eval_duration"] = distr_response[5]
        out_msg["eval_count"] = distr_response[6]
        out_msg["eval_duration"] = distr_response[7]

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
                        "name": "model_name",  <-- from checkpoints dir (org/name)
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
            - last_update: UNIX timestamp
            TODO
        """
        self.node_capabilities["node_config"] = self.node_config
        self.node_capabilities["role"] = self.role
        self.node_capabilities["model"] = {
            "active": self.gptdistr.model_type,
            "available": get_available_models(self.ckpt_dir),
        }
        self.node_capabilities["last_update"] = time.time()
        return self.node_capabilities

    # ----- REST API ------------------------------------------------------------------

    def DELETE(self):
        """Not implemented"""
        raise cp.HTTPError(501, "DELETE not implemented!")


class StarterServer(GPTServer):
    """TODO"""

    # Keep track of connected nodes (store their 'capabilities')
    all_nodes_registered_event = threading.Event()

    init_msg = {
        "role": "",
        "prev_node": {},
        "next_node": {},
        "model_config": {},
        "n_nodes": 0,
        "n_samples": 0,
        "max_seq_length": None,
    }

    def __init__(
        self,
        node_config: Dict[str, Any],
        node_type: str,  # In secondary nodes, uninitialized: "secondary"
        model: FileType,
        ckpt_dir: FileType = script_dir / ".." / "checkpoints",
        *,
        max_seq_length: Optional[int] = None,
        model_device: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            node_config,
            node_type,
            ckpt_dir,
            max_seq_length=max_seq_length,
            model_device=model_device,
            dtype=dtype,
            **kwargs,
        )

        self.own_config = self.node_config["nodes"]["starter"]
        self.secondary_list = self.node_config["nodes"]["secondary"]
        self.n_secondary = len(self.secondary_list)
        self.nodes_registry = [{}] * self.n_secondary
        self.registered_secondaries = [False] * self.n_secondary
        self.n_nodes = 1 + self.n_secondary

        if not model:
            raise ValueError("No model was specified!")

        self.model = model  # Format: organization/model_name
        self.model_org, self.model_name = str(self.model).split("/")
        self.model_dir = self.ckpt_dir / model
        self.chunk_path = get_chunk_path(
            self.ckpt_dir, self.model, self.n_nodes, self.role
        )

        if not self.model_dir.exists():
            raise NotADirectoryError(f"Unable to find directory for model {model}")
        if not len(os.listdir(self.model_dir)):
            raise FileNotFoundError("Model directory is empty!")

        # TODO: add support for downloading model as well (extra)

        # Determine whether model was split or not
        self.model_was_split = True
        self.chunk_path = get_chunk_path(
            self.ckpt_dir, self.model, self.n_nodes, self.role
        )
        self.full_model_path = get_chunk_path(self.ckpt_dir, self.model, 1, self.role)
        # Possible to run without full model stored, but only chunks
        if not self.chunk_path.exists():
            if self.n_nodes == 1:
                raise FileNotFoundError("Full model not found")
            else:
                if not self.full_model_path.exists():
                    raise FileNotFoundError("Full model not found!")
                self.model_was_split = False

        if not self.model_was_split:
            # Split model and store chunks
            self.model_config, full_model = load_from_pt(self.full_model_path)
            assert full_model is not None
            split_and_store(full_model, self.n_nodes, self.model_dir)
            del full_model
            full_model = None
            gc.collect()
        else:
            self.model_config = load_model_config(self.model_dir)

        self.n_local_layers = N_LAYERS_NODES[self.n_nodes][self.model_config.n_layer][
            "N_LAYERS_START"
        ]

        self.model_seq_length = max_seq_length
        if (
            self.model_seq_length
            and self.model_seq_length > self.model_config.block_size
        ):
            raise ValueError(
                f"The truncated sequence length {self.model_seq_length} should be "
                "lower or equal than the model's max sequence length "
                f"{self.model_config.block_size}"
            )

    def node_registration(self, capabilities):
        """
        Given node info, see if it matches one of the secondary nodes in 'node_config',
        and, if yes, add it to the 'nodes_registry'.

        A node is uniquely identified by its IP, communication port and inference ports,
        in other words, its "node_config"

        Args:
            capabilities: node capabilities of candidate

        Returns:
            1 if success
            0 if not expected
            -1 if already registered
        """
        if self.n_nodes == 1:
            return -1
        ind = 0
        while capabilities["node_config"] != self.secondary_list[ind]:
            ind += 1

        if ind < len(self.secondary_list):
            if self.registered_secondaries[ind]:
                return -1
            # Node is expected, add it (correct position)
            self.nodes_registry[ind] = capabilities
            self.registered_secondaries[ind] = True
            return 1
        else:
            return -1

    def wait_for_nodes_registration(self):
        """
        Wait until all required secondary nodes have been registered (PUT)
        """
        self.all_nodes_registered_event.wait()

    def init_model(self) -> bool:
        """
        Initialize starter model and tokenizer.

        Returns true upon successful initialization.
        """
        try:
            self.gptdistr.load_tokenizer(self.model_dir)
            self.gptdistr.init_model(self.n_local_layers, model_path=self.chunk_path)
            return True
        except Exception as e:
            print(e)
            return False

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
        self.inference_thread = threading.Thread(target=self.start_inference)

        # NOTE: the separate thread is just a placeholder to make the interface uniform
        # for all nodes - here we wait for the processing loop to conclude!
        # The thread in which this method is ran will stop here until processing stops
        self.inference_thread.start()
        self.inference_thread.join()
        self.shutdown()

    def get_nodes_info(self):
        """
        Read list of registered nodes.
        """
        return self.nodes_registry

    def configure_nodes(self) -> bool:
        """
        Send POST requests to the other nodes to inform them of their role, the number
        of samples, and including their chunk of model.

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
        assert self.n_nodes is not None
        assert self.chunk_path is not None
        node_chunks_dir = self.chunk_path.parent
        if not self.model_config:
            raise ValueError("The model configuration was not loaded!")
        if self.n_nodes != 1 and not self.all_nodes_registered_event.is_set():
            raise RuntimeError("Nodes have not registered yet!")

        self.get_nodes_info()

        # TODO: implement algorithm to select best set of nodes depending on owned
        # chunks

        out = True  # Return code
        # Store the prev and next in a smart way
        prev = self.node_config["nodes"]["starter"]
        if self.n_secondary == 1:
            next = self.node_config["nodes"]["starter"]
        elif self.n_secondary > 1:
            next = self.secondary_list[1]
        else:
            warnings.warn("No secondary nodes found! Running standalone")
            return out

        # Secondary nodes config
        for i, sec_node_cap in enumerate(self.nodes_registry):
            if self.verb:
                print(f"Initializing secondary node n.{i}")

            curr_msg = self.init_msg.copy()  # FIXME: maybe put before loop
            curr_msg["role"] = f"secondary:{i}"
            curr_msg["model_config"] = self.model_config.asdict()
            curr_msg["n_nodes"] = self.n_nodes
            curr_msg["n_local_layers"] = N_LAYERS_NODES[self.n_nodes][
                self.model_config.n_layer
            ]["N_LAYERS_SECONDARY"]
            curr_msg["prev_node"] = prev
            curr_msg["next_node"] = next
            curr_msg["max_seq_length"] = self.model_seq_length
            curr_msg["model_name"] = self.model_name

            # TODO: if secondary does not have model chunk, send it
            chunk_path = node_chunks_dir / f"model_secondary{i}.pth"
            if not is_model_chunk_available_secondary(
                str(self.model_name), i, self.n_nodes, sec_node_cap
            ):
                curr_msg["params"] = torch.load(chunk_path, device="cpu")

            # Update next and prev for next iteration
            prev = sec_node_cap
            if i == self.n_secondary - 1:  # Last iter in loop - finished
                next = None
            elif i == self.n_secondary - 2:  # Second to last iter
                next = self.node_config["nodes"]["starter"]
            else:
                next = self.node_config["nodes"]["secondary"][i + 2]

            # Send POST request
            target_addr = sec_node_cap["node_config"]["addr"]
            target_port = sec_node_cap["addr"]["communication"]["port"]

            addr = f"http://{target_addr}:{target_port}/init"
            out = out and (self._request_to_node("post", addr, curr_msg) is not None)

            if not out:
                if self.verb:
                    print("> Failed!")
                logger_wp.error(f"Failed to initialize secondary node {i}!")
                return out

            if self.verb:
                print("> Success!")
            logger_wp.info(f"Secondary node {i} was initialized successfully")

        return out

    def stop_nodes(self) -> int:
        """
        Send a PUT request to all nodes triggering the application interruption.
        """
        out = 1
        for sec_node in self.secondary_list:
            target_addr = sec_node["addr"]
            target_port = sec_node["communication"]["port"]

            addr = f"http://{target_addr}:{target_port}/stop"
            out *= self._request_to_node("put", addr, "")
        return out

    def _request_to_node(
        self,
        req_type: str,
        addr: str,
        content: Optional[Any] = None,
        max_n_requests: int = 100,
    ) -> Any:
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
        if req_type.lower() == "get":
            req_func = requests.get
        elif req_type.lower() == "post":
            req_func = requests.post
        elif req_type.lower() == "put":
            req_func = requests.put
        else:
            raise ValueError(f"Unsupported request type '{req_type}'")

        ret = None
        n_ret = 0
        if self.verb:
            print(f"Sending {req_type} request to {addr}")
            print(f"Payload: {len(pickle.dumps(content))} Bytes")
        try:
            # Specify timeout
            ret = req_func(
                addr,
                data=pickle.dumps(content) if content else None,
                timeout=100,
            )

            if ret.status_code == 413:
                raise ConnectionError(f"Max payload for {req_type} was exceeded!")
            logger_wp.debug(
                f"Successful {req_type} request sent to {addr} - code {ret.status_code}"
            )
        except requests.exceptions.Timeout:
            if self.verb:
                print("Connection timed out!")
            logger_wp.warning("Request timed out!")
            n_ret += 1
        except:
            logger_wp.warning(f"Unable to submit {req_type} request sent to {addr}")
            n_ret += 1
        while (ret is None or ret.status_code != 200) and n_ret < max_n_requests:
            if self.verb:
                print(
                    f"""Unable to reach node ({addr}) - retrying in 2s
                    ({n_ret}/{max_n_requests})"""
                )
            time.sleep(2)
            try:
                ret = req_func(
                    addr,
                    data=pickle.dumps(content),
                    timeout=10000,
                )
                logger_wp.debug(
                    f"""Successful {req_type} request sent to {addr} - code
                    {ret.status_code}"""
                )
            except requests.exceptions.Timeout:
                if self.verb:
                    print("Connection timed out!")
                logger_wp.warning(f"Request timed out!")
            except:
                logger_wp.warning(f"Unable to submit {req_type} request sent to {addr}")
            n_ret += 1

        if ret is not None and ret.status_code == 200:
            return ret.text
        return None

    def process_user_prompt(self, http_msg_body: Dict[str, Any]):
        """
        Entrypoint for the user - will be called by POST
        """
        # NOTE: for now, ignore the keys that are not "prompt"
        assert "model" in http_msg_body, "Missing 'model' key"
        new_prompt = http_msg_body["prompt"]
        if new_prompt == "":
            new_prompt = "\n"

        distr_resp = self.gptdistr.create_new_sample(new_prompt)
        self.n_samples = self.gptdistr.n_samples

        # Make sure to return something in here! (JSON resp)
        return self._build_serv_resp(http_msg_body, distr_resp)

    # ------------ REST ---------------------------------------------------------------

    def GET(self, *path):
        """
        /: return node capabilities
        /registered_nodes: json containing registered secondary nodes
        """
        if not len(path):
            return json.dumps(self._get_node_capabilities())
        else:
            if str(path[0]) == "registered_nodes":
                return json.dumps(self.nodes_registry)
            else:
                raise cp.HTTPError(404, "Not found")

    def POST(self, *path):
        """
        Ollama-like APIs.

        TODO: allow to select model through API - will need to
        """
        if not len(path):
            raise cp.HTTPError(404, "Not found")
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
        else:
            raise cp.HTTPError(404, "Not found")

    def PUT(self, *path):
        """
        Interface for secondary node registration.
        """
        if path[0] == "register":
            content_type = cp.request.headers["Content-Type"]
            raw_body = cp.request.body.read().decode("utf-8")
            if "application/json" not in content_type:
                return "Unsupported Media Type", 415
            body = json.loads(raw_body)  # Should be node_capabilities
            out = self.node_registration(body)
            # If all nodes have registered
            if len(self.nodes_registry) >= len(self.secondary_list):
                self.all_nodes_registered_event.set()
            if out == 1:
                cp.response.status = 200
                return "Node registered successfully"
            elif out == -1:
                raise cp.HTTPError(400, "Node already registered!")
            else:  # == 0
                raise cp.HTTPError(400, "Node not expected!")
        else:
            raise cp.HTTPError(404, "Not found")


class SecondaryServer(GPTServer):
    """TODO"""

    registered_at_starter = False

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
        super().__init__(
            node_config,
            node_type,
            ckpt_dir,
            max_seq_length=max_seq_length,
            model_device=model_device,
            dtype=dtype,
            **kwargs,
        )
        self.secondary_index = -1  # Not init

    # ---------------------------------------------------------------------------------

    def register_at_starter(self):
        """
        Send PUT request to starter node containing capabilities.
        """
        if not self.role != "starter":
            raise AttributeError(
                "`register_at_starter` can only be called on secondary nodes"
            )
        # TODO
        pass

    # ------------ REST ---------------------------------------------------------------

    def GET(self, *path):
        """
        Functions
            Return node information (capabilities).
            Can be used for pinging "neighbor" nodes.
        """
        if not len(path):
            return json.dumps(self._get_node_capabilities())
        else:
            raise cp.HTTPError(404, "Not found")

    def POST(self, *path):
        """
        Functions:
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
            self.role is None or "secondary" in self.role
        ) and self.model is None:  # Only for non-init nodes
            if len(path) > 0 and path[0] == "init":
                assert not self.running.is_set()
                init_msg = pickle.loads(cp.request.body.read())
                if self.role is None:
                    self.role = init_msg["role"]
                    self.secondary_index = int(self.role.split(":")[1])
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

                # TODO: Read model and role, and determine whether chunk is owned
                # If not, make sure model chunk was sent
                # NOTE: secondary node should not have self.model_path (only
                # checkpoints directory)
                self.model_name = init_msg["model_name"]
                self.node_capabilities = self._get_node_capabilities()  # Updates them

                # Determine whether to expect the chunk from the starter or not
                chunk_expected = not is_model_chunk_available_secondary(
                    self.model_name,
                    self.secondary_index,
                    self.n_nodes,
                    self.node_capabilities,
                )

                model_org, model_name = self.model_name.split("/").apply()
                self.model_path = (
                    self.ckpt_dir
                    / model_org
                    / model_name
                    / "chunks"
                    / f"{self.n_nodes}nodes"
                    / f"model_secondary{self.secondary_index}.pth"
                )
                if chunk_expected:
                    assert (
                        "params" in init_msg
                    ), "Missing model chunk parameters from starter node!"
                    if self.verb:
                        print("Received parameters from starter")
                    chunk_sd = init_msg["params"]
                    # TODO: save chunk in local filesystem - handle lack of space
                    n_layers_detect = count_transformer_blocks(chunk_sd)
                    self.gptdistr.init_model(
                        self.n_layers_local, model_parameters=chunk_sd
                    )

                    # Free memory
                    del chunk_sd
                    chunk_sd = None
                    gc.collect()
                else:
                    if self.verb:
                        print("Loading parameters from disk")
                    self.gptdistr.init_model(
                        self.n_layers_local, model_path=self.model_path
                    )

                if VERB:
                    print(f"[INFO] Starting operation - {self.role} node")

                self.inference_thread = threading.Thread(
                    target=self.start_inference, daemon=True
                )
                self.inference_thread.start()
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found")
        else:
            raise cp.HTTPError(404, "Not found")

    def PUT(self, *path):
        """
        Used by the starter to stop running nodes at the end of the generation.

        Note: NOT used for terminating individual sample, but for interrupting run and
        releasing model.

        TODO: how to stop application??????
        """
        if self.role == "starter":
            pass
        else:
            # TODO: fix shutdown procedure
            if len(path) > 0 and path[0] == "stop":
                self._end_thr = threading.Thread(target=self.shutdown)
                self._end_thr.start()
                # self._end_thr.join()  # cannot wait, since thread stops server
                if self.verb:
                    print("[INFO] Node stopped through PUT request!")
                logger_wp.info("Received stopping directive")
                cp.response.status = 200
            else:
                raise cp.HTTPError(404, "Not found!")
