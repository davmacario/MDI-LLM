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
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Union

import cherrypy as cp

from sub.gptserver import SecondaryServer, StarterServer
from sub.utils.typing import FileType

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

    n_nodes: Optional[int] = None

    def __init__(
        self,
        node_type: str,
        node_config: Union[Dict, FileType],
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

        self.role = node_type
        # NOTE: node_config can either be path or opened dict!!!
        if isinstance(node_config, FileType):
            self.config_file_path = Path(node_config)
            with open(self.config_file_path, "r") as f:
                self.node_config = json.load(f)
        elif isinstance(node_config, dict):
            self.config_file_path = None
            self.node_config = node_config

        if self.verb:
            print("[INFO] Loaded nodes config file")

        if self.role == "starter":
            if not model:
                # FIXME: in first version, only use model specified at initialization
                # Next versions will request model on-demand (with POST, like Ollama)
                raise ValueError("No model was specified!")

            self.model = model  # Format: organization/model_name
            self.model_dir = self.ckpt_dir / model
            if not self.model_dir.exists():
                raise NotADirectoryError(f"Unable to find directory for model {model}")
            if not len(os.listdir(self.model_dir)):
                raise FileNotFoundError("Model directory is empty!")

            self.n_secondary = len(self.node_config["nodes"]["secondary"])
            self.n_nodes = 1 + self.n_secondary
            if self.verb and self.n_nodes == 1:
                print("[INFO] Running in standalone mode!")
            self.own_config = self.node_config["nodes"]["starter"]
            self.own_addr = self.own_config["addr"]
            self.own_comm_port = self.own_config["communication"]["port"]
            self.own_inference_port_in = self.own_config["inference"]["port_in"]
            self.own_inference_port_out = self.own_config["inference"]["port_out"]

            self.gpt_serv = StarterServer(
                node_config=self.node_config,
                node_type=self.role,
                model=self.model,
                ckpt_dir=self.ckpt_dir,
                max_seq_length=model_seq_length,
                model_device=self.torch_device,
                dtype=dtype,
                **kwargs,
            )

        elif "secondary" in self.role:
            # NOTE: the node_config should be the JSON spec of the node only
            # Initialize secondary node (we can't know in advance the index)
            self.own_config = self.node_config
            self.own_addr = self.own_config["addr"]
            self.own_comm_port = self.own_config["communication"]["port"]
            self.own_inference_port_in = self.own_config["inference"]["port_in"]
            self.own_inference_port_out = self.own_config["inference"]["port_out"]

            self.gpt_serv = SecondaryServer(
                node_config=self.node_config,
                node_type=self.role,
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
        if self.role == "starter":
            assert isinstance(self.gpt_serv, StarterServer)
            # Wait for node registration
            self.gpt_serv.wait_for_nodes_registration()
            # Init. nodes, launch iterations
            if not self.gpt_serv.configure_nodes():
                raise RuntimeError("Unable to initialize network nodes!")
            if not self.gpt_serv.init_model():
                # TODO - expecting init_model to be similar to configure_nodes
                raise RuntimeError("Unable to initialize starter node")

            try:
                self.gpt_serv.launch_starter()
            except KeyboardInterrupt:
                self.gpt_serv.shutdown()
                self.gpt_serv.stop_nodes()
                print("Node was stopped!")
        else:
            assert isinstance(self.gpt_serv, SecondaryServer)
            self.gpt_serv.register_at_starter()
            try:
                cp.engine.block()  # Same as while True: time.sleep(...)
            except KeyboardInterrupt:
                self.gpt_serv.shutdown()
                print("Node was stopped!")
