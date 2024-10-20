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

import json
import logging
import os
import pickle
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import cherrypy as cp
import requests
import torch

from sub.config import N_LAYERS_NODES
from sub.gptserver import GPTServer
from sub.utils import load_from_pt, load_model_config, split_and_store
from sub.utils.typing import FileType, JSONType

docstring = """
Distributed implementation of the Llama architecture using Model-Distributed Inference
(with pipeline parallelism).
This implementation allows to run a Llama model (of the ones compatible with LitGPT)
over a network of "nodes" that can be positioned on different physical hosts.

The distributed implementation consists of partitioning the model layers among all the
nodes in the network. Then, each node will work with a subset of layers, receiving the
input from the previous node in the network, and transmitting the output to the next
one.
This allows to reduce the memory usage for each node compared to the memory required to
run the complete model on a single node, allowing to run larger models by increasing the
number of network nodes.

Computation parallelism is introduced by increasing the batch size, allowing to generate
different independent samples.
Due to the autoregressive nature of LLMs, to generate a new token in the sequence, it is
required to feed back the new token to the model input, hence, if generating a single
piece of text, the nodes that are not currently busy processing their local model chunk
would be idle, waiting for the information to reach them.
If we generate more than one sample, it is possible for different nodes to work on
different samples concurrently, improving efficiency.
In particular, when the number of samples (batch size) is greater or equal than the
number of nodes, it is possible to ensure every node is always working on a different
sample.
This mechanism, that we call "recurrent pipelining", allows to achieve a generation rate
(tokens/second) which is higher than sequential generation on a single device.
With this work, we provide a proof of concept for the use of pipeline parallelism for
transformer architecture inference, resulting in an optimized implementation achieving
competitive performances, with the added bonus of enabling LLM deployment on Edge
devices (the testbed for this project was made up of Nvidia Jetson TX2 modules).

The application architecture is the following.
We define 2 types of nodes: "starter" nodes and "secondary" nodes; unlike the name
suggests, there is no "master-slave" relationship between the 2, the "starter" is just
acting as the "entrypoint"/application interface with the user and actively participates
in the computation.
Starter nodes are used to initialize secondary nodes and contain the first and last
layers of the model. They take the model input (prompt) and collect the output tokens.
As a consequence, they are the ones that "know" the exact number of tokens (and
iterations) to be performed in the current inference run.
Secondary nodes are "activated" by the starting node, and just receive inputs to be
passed through the local model chunk.
For efficiency reasons, we assume the model chunks are already located on the devices
themselves, but it is also possible to have the starter node split the model layers and
send them to the different devices.

Communication happens over 2 channels.
For coordination and initialization, each node acts as an HTTP server and messages are
sent over HTTP.
For the transmissions of intermediate activations, the nodes use bare Python sockets
running over TCP/IP. The lack of a fully-fledged application layer protocol allows for
a faster message exchange.
At the application layer, the message only contains a header of fixed length, specifying
the exact message size in bytes, which allows to read the exact amount of bytes to
prevent issues due to message truncation.
Being message transmission a crucial part of the application, as it is necessary to
ensure it does not slow down the overall operation, we implement it through input and
output message queues running on separate threads.
Once a message is received, it is placed in the input queue, from where the processing
thread (the one performing model forward passes) will extract it (in order), process it,
and place the output message in the output queue.
There, a separate thread will extract it and transmit it.

The application is composed of the following modules:
- App: entrypoint for initializing nodes of any type; for starter nodes, it
provides methods used at initialization.
- GPTServer: core of the application; it creates the HTTP server for coordination and
sets up the message transmission sockets. It contains the definition of all the
application threads and the processing loop.
- GPTDistributed: model definition, based on LitGPT (by Lightning AI, in turn based on
NanoGPT).
The actual application uses submodels over with the same architecture (with the same
building blocks).
"""

script_dir = Path(os.path.dirname(__file__))

logger_wp = logging.getLogger("model_dist")
logger_wp.setLevel(logging.ERROR)

MODEL_TYPE = ""
CTX = nullcontext()


class App:
    __doc__ = docstring
    init_msg = {
        "role": "",
        "prev_node": {},
        "next_node": {},
        "model_config": {},
        "n_nodes": 0,
        "n_samples": 0,
        "max_seq_length": None,
    }

    secondary_nodes_config: List[JSONType] = []

    n_nodes: Optional[int] = None

    def __init__(
        self,
        node_type: str,
        node_config: FileType,
        *,
        ckpt_dir: FileType = script_dir / ".." / "checkpoints",
        model: Optional[FileType] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        model_seq_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Instantiate a GPTDistributed object, allowing to run a node for
        Model-Distributed Inference.

        Args:
            node_type: role of the node - can be "starter" or "secondary".
            node_config: path of the configuration file for the node OR dict containing
                the configuration itself;
                In starters: it is the configuration of the whole network (full
                    config.json).
                In secondary: it is the full config or the individual node config.
            *
            ckpt_dir: optional path containing all the models subdir; defaults to
                `./../checkpoints`
            model: (WILL BE REMOVED) [starter only] model directory path relative to
                ckpt_dir; if not found, it will be downloaded.
            device (default: None): string indicating the device used to load and run
                the model; if not specified, the application will try to get it from the
                nodes configuration file.
            dtype: (optional) data type to be used; if missing, will use default one.
            model_seq_length (optional): maximum sequence length of the model; should be
                less or equal than the one specified in the config (default value)
            Keyword args (optional): allowing to specify verb=self.verb and plots=PLOTS
                (bool)
        """
        self.ckpt_dir = Path(ckpt_dir)

        self.torch_device = device if device else None

        self.verb = False if "verb" not in kwargs else bool(kwargs["verb"])
        self.plots = False if "plots" not in kwargs else bool(kwargs["plots"])

        self.compile = False if "compile" not in kwargs else bool(kwargs["compile"])
        self.dtype = dtype

        self.node_type = node_type
        # NOTE: node_config can either be path or opened dict!!!
        if isinstance(node_config, FileType):
            self.config_file_path = Path(node_config)
            with open(self.config_file_path, "r") as f:
                self.node_config = json.load(f)
        elif isinstance(node_config, dict):
            self.config_file_path = None
            self.node_config = node_config

        if self.verb:
            print("Loaded nodes config file")

        if self.node_type == "starter":
            if not model:
                # FIXME: in first version, only use model specified at initialization
                # Next versions will request model on-demand (with POST, like Ollama)
                raise ValueError("No model was specified!")

            self.model_dir = self.ckpt_dir / model
            if not self.model_dir.exists():
                raise NotADirectoryError(f"Unable to find directory for model {model}")
            if not len(os.listdir(self.model_dir)):
                raise FileNotFoundError("Model directory is empty!")

            self.n_secondary = len(self.node_config["nodes"]["secondary"])
            self.n_nodes = 1 + self.n_secondary
            if self.verb and self.n_nodes == 1:
                print("Running in standalone mode!")
            self.own_config = self.node_config["nodes"]["starter"]
            self.own_addr = self.own_config["addr"]
            self.own_comm_port = self.own_config["communication"]["port"]
            self.own_inference_port_in = self.own_config["inference"]["port_in"]
            self.own_inference_port_out = self.own_config["inference"]["port_out"]

            # TODO: add support for downloading model as well (extra)

            # Determine whether model was split or not
            self.model_was_split = True
            if self.n_nodes > 1:
                node_chunks_dir = self.model_dir / f"{self.n_nodes}nodes"
                self.chunk_path = node_chunks_dir / "model_starter.pth"
                if not self.chunk_path.exists():
                    self.model_was_split = False
            else:
                self.chunk_path = self.model_dir / "lit_model.pth"

            # Load model config
            self.model_config = load_model_config(self.model_dir)

            self.model_seq_length = None
            if model_seq_length and model_seq_length > self.model_config.block_size:
                raise ValueError(
                    f"The truncated sequence length {model_seq_length} should be lower "
                    "or equal than the model's max sequence length "
                    f"{self.model_config.block_size}"
                )
            else:
                self.model_seq_length = model_seq_length

            self.gpt_serv = GPTServer(
                node_config=self.node_config,
                node_type=self.node_type,
                ckpt_dir=self.ckpt_dir,
                model_seq_length=self.model_seq_length,
                model_device=self.torch_device,
                dtype=dtype,
                **kwargs,
            )

        elif "secondary" in self.node_type:
            # NOTE: the node_config should be the JSON spec of the node only
            # Initialize secondary node
            self.own_config = self.node_config
            self.own_addr = self.own_config["addr"]
            self.own_comm_port = self.own_config["communication"]["port"]
            self.own_inference_port_in = self.own_config["inference"]["port_in"]
            self.own_inference_port_out = self.own_config["inference"]["port_out"]

            self.gpt_serv = GPTServer(
                node_config=self.node_config,
                node_type=self.node_type,
                ckpt_dir=self.ckpt_dir,
                model_device=self.torch_device,
                dtype=dtype,
                **kwargs,
            )

        # Here because if the 'device' arg is None, gpt_serv will infer it
        self.torch_device = self.gpt_serv.model_device

        # Webserver started ASAP
        self.gpt_serv.start_webserv()

    def start(self):
        """
        Main class entrypoint.
        Start the application; for the starter node, this triggers the initialization of
        other nodes and launches the web server awaiting for generation.
        For secondary nodes, this starts an infinite loop where the node will wait to be
        initialized and perform inference.
        """
        if self.node_type == "starter":
            assert self.model_config
            # Wait for node registration
            self.gpt_serv.wait_for_nodes_registration()
            # Init. nodes, launch iterations
            if not self.configure_nodes():
                raise RuntimeError("Unable to initialize network nodes!")

            try:
                self.gpt_serv.launch_starter()
            except KeyboardInterrupt:
                self.gpt_serv.shutdown()
                self.stop_nodes()
                print("Node was stopped!")
        else:
            try:
                cp.engine.block()  # Same as while True: time.sleep(...)
            except KeyboardInterrupt:
                self.gpt_serv.shutdown()
                print("Node was stopped!")

    # ---------------------------------------------------------------------------------

    def get_nodes_info(self):
        """
        Send GET to all nodes listed in the config JSON to obtain the information about
        the nodes and the available models (and chunks).

        TODO: design algorithm to look for a good node combination so that the nodes
        contain all the model
        """
        for node in self.node_config["nodes"]["secondary"]:
            addr = f"http://{node['addr']}:{node['communication']['port']}/"
            self.secondary_nodes_config.append(
                json.loads(self._request_to_node("get", addr))
            )

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
        if self.node_type != "starter":
            raise ValueError("This method can only be called on starter nodes!")
        if not self.model_config:
            raise ValueError("The model configuration was not loaded!")

        self.get_nodes_info()

        # TODO: implement algorithm to select best

        out = True  # Return code
        # Store the prev and next in a smart way
        prev = self.node_config["nodes"]["starter"]
        if self.n_secondary == 1:
            next = self.node_config["nodes"]["starter"]
        elif self.n_secondary > 1:
            next = self.node_config["nodes"]["secondary"][1]
        else:
            warnings.warn("No secondary nodes found! Running standalone")
            return out

        # Secondary nodes config
        for i, sec_node in enumerate(self.node_config["nodes"]["secondary"]):
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

            if not self.model_was_split:
                chunk_path = node_chunks_dir / f"model_secondary{i}.pth"
                curr_msg["params"] = torch.load(chunk_path, device="cpu")

            # Update next and prev for next iteration
            prev = sec_node
            if i == self.n_secondary - 1:  # Last iter in loop - finished
                next = None
            elif i == self.n_secondary - 2:  # Second to last iter
                next = self.node_config["nodes"]["starter"]
            else:
                next = self.node_config["nodes"]["secondary"][i + 2]

            # Send POST request
            target_addr = sec_node["addr"]
            target_port = sec_node["communication"]["port"]

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
        for sec_node in self.node_config["nodes"]["secondary"]:
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
            logger_wp.warning(f"Request timed out!")
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
