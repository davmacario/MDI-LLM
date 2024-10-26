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

import accelerate
import cherrypy as cp
import torch
from torch.cuda import OutOfMemoryError

from sub.config import DEVICE as DEFAULT_DEVICE
from sub.config import (DTYPE, DTYPE_TORCH_MAPPING, N_LAYERS_NODES,
                        TEMPERATURE, TOP_K, VERB)
from sub.connections import InputNodeConnection, OutputNodeConnection
from sub.model import Config, KVCache, sample
from sub.prompts import PromptStyle, has_prompt_style, load_prompt_style
from sub.submodels import SecondaryNode, StarterNode
from sub.tokenizer import Tokenizer
from sub.utils import (catch_loop_errors, count_transformer_blocks,
                       detect_stop_tokens, find_eot, get_available_models,
                       load_sd, s_to_ns, waiting_animation)
from sub.utils.typing import FileType, JSONObject, JSONType

script_dir = Path(os.path.dirname(__file__))


class GPTDistributed:
    verb = False

    own_config: Dict[str, Any] = {}

    model: Optional[Union[StarterNode, SecondaryNode]] = None
    next_node: Optional[Dict] = None
    prev_node: Optional[Dict] = None
    model_params: Optional[Dict] = None
    model_config: Optional[Config] = None
    model_type: str = ""  # Active model - "" if no initialized model

    # Number of samples that have been requested so far in the current run:
    n_samples: int = 0
    # Map sample ID to n. of iteration - initialized to 0 when new sample is created
    iter_ind: Dict[int, int] = {}

    T_i: Dict[int, int] = {}  # Contains size of context of each prompt
    # Will contain the input pos. of all samples:
    input_pos: Dict[int, torch.Tensor] = {}
    kvcaches: Dict[int, List[KVCache]] = {}

    samples: Dict[int, torch.Tensor] = {}
    prompt_lengths: Dict[int, int] = {}
    max_new_tokens: Dict[int, int] = {}
    sample_finished: Dict[int, threading.Event] = {}

    # QUEUES: should be mapped to queues in GPTServer to provide callbacks
    # If None, need to initialize them!
    in_message_queue: Optional[deque] = None
    in_queue_not_empty: Optional[threading.Event] = None
    out_message_queue: Optional[deque] = None
    out_queue_not_empty: Optional[threading.Event] = None

    # Set iff the model has been initialized and it is ready to perform inference.
    running = threading.Event()

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

    # Response queue - used to pass generated responses from loop to HTTP server
    resp_msg_template = {}  # TODO: use Ollama's syntax
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

    def __init__(
        self,
        node_config: Dict[str, Any],
        node_type: str,
        ckpt_dir: FileType = script_dir / ".." / "checkpoints",
        *,
        max_seq_length: Optional[int] = None,
        model_device: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize GPTDistributed object.

        This object will control a specific model (Starter/Secondary), allowing to pass
        on the information in the chain while performing inference.

        The node will be initialized given the configuration

        Args:
            node_config: node configuration information (from .json file)
            node_type: string indicating the node type/role ("starter" or "secondary");
                for secondary nodes, the role will be updated to "secondary:<n>" after
                receiving the initialization from the starter.
            ckpt_dir: directory in which the models are found; subdirs of this path are
                <org>/<model_name>.
            *
            max_seq_length: maximum sequence length of the model, allows to specify a
                value shorter than the actual one in order to reduce (V)RAM usage
            model_device: device where to load the model chunk; can be omitted if
                specified in the node_config (arg will override it!)
            dtype: string indicating the torch data type ("float16", "float32" or
                "bfloat16")
            **kwargs: support for 'verb' and 'compile' (torch>=2.0)
        """
        # Override global constants with kwargs
        self.verb = VERB if "verb" not in kwargs else kwargs["verb"]
        self.compile = False if "compile" not in kwargs else kwargs["compile"]

        self.node_config = node_config
        self.ckpt_dir = Path(ckpt_dir)
        self.role = node_type

        self.max_seq_length = max_seq_length  # If None, will use model's default

        # Select device (fills self.model_device [str] and self.torch_device [torch])
        self._select_device(model_device)
        assert isinstance(self.model_device, str) and self.torch_device

        # Dtype initialization - use default DTYPE if not given
        self._select_dtype(dtype)
        assert isinstance(self.dtype, str) and self.ptdtype

        if "starter" in self.role:
            # The node_config for the starter is the WHOLE JSON! It should know the
            # other nodes in the network to initialize them
            self.role = "starter"
            self.own_config = node_config["nodes"]["starter"]

            self.n_nodes = 1 + (
                0
                if "secondary" not in node_config["nodes"]
                else len(node_config["nodes"]["secondary"])
            )

            if self.n_nodes == 1:
                self.out_message_queue = self.in_message_queue
                self.out_queue_not_empty = self.in_queue_not_empty

            # TODO: figure out where starter model is initialized
            # OLD: self._init_model()
        else:
            # NOTE: index of secondary will be inferred from POST for initialization
            # NOTE: `node_config` for secondary node only contains its own info
            self.own_config = node_config
            self.starter_addr = self.own_config["communication"]["starter_addr"]

        # Init own info
        self.own_addr = self.own_config["addr"]
        self.own_comm_port = self.own_config["communication"]["port"]
        self.inference_port_in = self.own_config["inference"]["port_in"]
        self.inference_port_out = self.own_config["inference"]["port_out"]

    # ---------------------------------------------------------------------------------

    def create_new_sample(self, new_prompt):
        if not self.prompt_style:
            raise RuntimeError("Prompt style has not been initialized")
        if not self.tok:
            raise RuntimeError("Tokenizer has not been initialized")
        if not self.model:
            raise RuntimeError("Model has not been initialized")
        if not self.are_queues_init():
            raise RuntimeError("Queues are not initialized!")
        if not self.role == "starter":
            raise ValueError("Cannot call method on secondary nodes")

        # Keep track of time
        t_start_tot_ns = s_to_ns(time.time())

        start_styled = self.prompt_style.apply(new_prompt)

        # Create new sample by processing prompt
        new_idx = self.tok.encode(start_styled, device=self.torch_device).view(1, -1)
        t_prompt_eval = s_to_ns(time.time()) - t_start_tot_ns

        new_id = self.n_samples
        self.n_samples += 1
        if self.verb:
            print(f"Created new sample {new_id}")

        # NOTE: self.samples[new_id] will be initialized in the loop
        prompt_len = len(new_idx.squeeze())
        self.prompt_lengths[new_id] = prompt_len
        self.iter_ind[new_id] = 0
        self.sample_finished[new_id] = threading.Event()
        self.sample_finished[new_id].clear()
        self.max_new_tokens[new_id] = (
            self.model.max_seq_length - self.prompt_lengths[new_id]
        )
        if self.max_new_tokens[new_id] <= 0:
            raise RuntimeError(
                f"Prompt for sample {new_id} is longer than the model sequence length"
            )

        # Place sample in input queue
        t_start_load_ns = s_to_ns(time.time())
        self.in_message_queue.append(self._build_msg(new_idx, new_id))
        self.in_queue_not_empty.set()

        # Retrieve response - loop should signal end (e.g., Event)
        self.sample_finished[new_id].wait()
        out_sample_tensor = self.samples[new_id]

        # FIXME: redundant
        out_truncated = find_eot(
            out_sample_tensor, self.stop_tokens, self.prompt_lengths[new_id]
        )
        n_out_tokens = len(out_truncated.squeeze())
        out_truncated_no_prompt = out_truncated[0][prompt_len:]

        if self.verb:
            print(
                f"Truncated sample {new_id} to {n_out_tokens}/"
                f"{len(out_sample_tensor.squeeze())}"
            )
        t_start_decode = s_to_ns(time.time())
        gen_text = self.tok.decode(out_truncated_no_prompt)

        t_stop_ns = s_to_ns(time.time())
        t_tot_ns = t_stop_ns - t_start_tot_ns
        t_load_ns = t_stop_ns - t_start_load_ns
        t_decode = t_stop_ns - t_start_decode

        # Clear starter's caches
        self._delete_starter_caches(new_id)

        # FIXME: choose better format
        return (
            gen_text,
            new_id,
            t_tot_ns,
            t_load_ns,
            prompt_len,
            t_prompt_eval,
            n_out_tokens,
            t_decode,
        )

    def init_msg_queues(
        self,
        q_in: deque,
        q_out: deque,
        q_in_not_empty: threading.Event,
        q_out_not_empty: threading.Event,
    ):
        """
        This method should be called by the GPTServer to provide callbacks for the msg
        queues and the related events.

        In the architecture, the message handling is done by GPTServer, but
        GPTDistributed still needs a way to place/retrieve messages in/from the queues.
        This method is used to provide the queues.

        Args:
            q_in: input deque
            q_out: output deque
            q_in_not_empty: Event that is set when the input queue contains >= 1 msg
            q_out_not_empty: Event that is set when the output queue contains >= 1 msg
        """
        self.in_message_queue = q_in
        self.in_queue_not_empty = q_in_not_empty

        self.out_message_queue = q_out
        self.out_queue_not_empty = q_out_not_empty

    def are_queues_init(self) -> bool:
        """Returns true if the message queues have been initialized already."""
        return (
            self.in_message_queue is not None
            and self.in_queue_not_empty is not None
            and self.out_message_queue is not None
            and self.out_queue_not_empty is not None
        )

    def init_model(
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
            n_transf_layers: number of transformer layers of the local model; required
                for initializing the submodel.
            *
            model_path: if present, it is the path in the local file system where
                the model chunk is found
            model_parameters: alternatively, the model parameters may already be present
                in memory (e.g., loaded previously or received by starter node)
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
        if self.verb:
            print("Initializing empty local model")

        self.n_layers_local = n_transf_layers
        Model_class = StarterNode if "starter" in self.role else SecondaryNode
        try:
            self.model = Model_class(self.model_config, n_transf_layers).to_empty(
                device=self.model_device
            )
        except OutOfMemoryError:  # FIXME: may not be right error
            with accelerate.init_empty_weights():
                self.model = Model_class(self.model_config, n_transf_layers).to("meta")

        assert self.model is not None

        if self.verb:
            print("Loading parameters")
            print(f"Using dtype {self.ptdtype}")

        # TODO: switch to "empty" models (torch >= 2.0)
        if model_parameters:
            # NOTE: if here, cannot use accelerate! Weights are already in memory...
            model_parameters = {
                k: v.to(self.ptdtype) for k, v in model_parameters.items()
            }
            # Check the model chunk contains the expected number of transformer layers
            n_layers_detect = count_transformer_blocks(model_parameters)
            if self.n_layers_local != n_layers_detect:
                raise ValueError(
                    f"The number of detected transformer blocks ({n_layers_detect}) is "
                    f"different from the expected one {self.n_layers_local}; please "
                    "re-run the model partition or check the configuration!"
                )
            self.model.load_weights(model_parameters)
            self.model = self.model.to(self.ptdtype)
        else:
            assert model_path
            # NOTE: using accelerate to prevent memory usage spike at the beginning
            # resulting from the need to load both the "empty" model and the weights
            self.model = accelerate.load_checkpoint_and_dispatch(
                self.model, str(model_path), dtype=self.ptdtype
            )
            # FIXME: cannot do sanity check on number of model layers (model not in mem)

        assert self.model, "The model was not correctly created"

        if self.max_seq_length:
            print(f"[DEBUG] Truncating context length to {self.max_seq_length}")
            self.model.max_seq_length = self.max_seq_length
        else:
            # Use default value
            self.max_seq_length = self.model.max_seq_length

        if self.verb:
            print(f"Moving model to {self.torch_device}")
        self.model = self.model.to(self.torch_device)

        if self.compile and hasattr(torch, "compile"):
            if self.verb:
                print("Compiling local model - this may take a while", end="\r")
            try:
                self.model = torch.compile(self.model)
                if self.verb:
                    print("Model compiled!                                ")
            except RuntimeError as e:
                warnings.warn(f"Unable to compile model! {e}")
        elif self.compile and not hasattr(torch, "compile"):
            from importlib.metadata import version

            warnings.warn(
                f"Installed torch version ({version('torch')}) does not support "
                "compiling models"
            )

        del model_parameters
        gc.collect()

        # Model is "ready" to be used - set running
        self.running.set()

    def load_tokenizer(self, tokenizer_dir: FileType):
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
        if self.verb:
            print("Loading tokenizer", end="")
        # FIXME: this is just to give priority to HF; some tokenizer_config.json files
        # (Llama 2) are broken...
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

        if self.verb:
            print(f"Prompt style: {type(self.prompt_style)}")

        self.stop_tokens = self.prompt_style.stop_tokens(self.tok)
        if self.verb:
            print("Tokenizer and prompt style have been loaded!")

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
            warnings.warn(
                f"Using default device {DEFAULT_DEVICE} - no device was "
                "specified through CLI or config JSON file"
            )
            self.model_device = DEFAULT_DEVICE
        self.torch_device = torch.device(self.model_device)
        if self.verb:
            print(f"Using device: {self.model_device}")

    def _select_dtype(self, dtype):
        """
        Select model data type.
        Priority:
            1. CLI arg (`--dtype`)
            2. Loaded model dtype
            3. Default dtype

        Important notice: the loaded model dtype will be found out at model
        initialization.
        """
        self.use_default_dtype = dtype is None
        self.dtype = dtype if dtype else DTYPE  # String
        if self.dtype == "bfloat16" and (
            not torch.cuda.is_available()
            or not torch.cuda.is_bf16_supported()
            or "cuda" not in self.model_device  # If using bfloat16, should use cuda
        ):
            raise ValueError(
                "Specified bfloat16, but the host does not support this format"
            )
        self.ptdtype = DTYPE_TORCH_MAPPING[self.dtype]  # torch dtype

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

    def _init_sample_caches(self, id, idx):
        """
        Initialize the model cache for the new sample `idx` with ID: `id`, using a
        specified dtype.

        Args:
            id: sample ID
            idx: new sample (encoded prompt)
            dtype: desired dtype for the KV caches
        """
        assert self.model is not None

        self.T_i[id] = idx.size(1)
        self.input_pos[id] = torch.arange(0, self.T_i[id], device=self.torch_device)
        kvc_sublist: List[KVCache] = []
        for _, block in enumerate(self.model.transformer.h):
            # Build kv cache individually for each attn layer
            kvc_sublist.append(
                block.attn.build_kv_cache(
                    batch_size=1,
                    max_seq_length=self.model.max_seq_length,
                    rope_cache_length=self.model.cos.size(-1),
                    device=self.torch_device,
                    dtype=self.ptdtype,
                )
            )
        self.kvcaches[id] = kvc_sublist

    def _delete_sample_caches(self, id):
        """
        Delete the cached parameters for sample `id`.

        Args:
            id: sample ID
        """
        if self.verb:
            print("[DEBUG] Releasing caches")
        # FIXME: maybe try-except to prevent issues if key not found for some reason
        # Find a clean way, as try-except should be done for all vars
        del self.T_i[id]
        del self.input_pos[id]
        del self.kvcaches[id]

    def _delete_starter_caches(self, id):
        """
        Used to delete the starter-exclusive caches (samples, iteration indices, prompt
        lengths, max_new_tokens and sample finished event) for a specific sample ID
        `id`.

        This method should only be called once the sample has been delivered to the user
        and the info is not needed anymore.
        """
        if "starter" not in self.role:
            warnings.warn("This method should only be called on starter nodes!")
        else:
            del self.samples[id]
            del self.iter_ind[id]
            del self.prompt_lengths[id]
            del self.max_new_tokens[id]
            del self.sample_finished[id]

    # ---- Main Loops -----------------------------------------------------------------

    def starter_loop(self) -> None:
        """
        Generation loop for the starter node only.
        """
        assert self.model_config and self.model
        assert self.model_device is not None
        assert self.in_message_queue is not None and self.in_queue_not_empty
        assert self.out_message_queue is not None and self.out_queue_not_empty

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

        # Initialize RoPE cache and attention mask
        self.model.init_rope_mask(device=self.torch_device)
        self.model.eval()

        # Starter Only
        self.samples: Dict[int, torch.Tensor] = {}
        self.prompt_lengths: Dict[int, int] = {}

        event_stop = threading.Event()
        loading_thread = threading.Thread(
            target=waiting_animation, args=("Processing samples", event_stop)
        )

        # TODO: add time logging
        start_time = time.time()

        if self.verb:
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

                    sample_id = in_msg["sample_index"]
                    if in_msg["stop"]:
                        # The stopping message made the whole loop - just ignore as
                        # caches should have already been cleared
                        pass
                    else:
                        idx = in_msg["data"].to(self.model_device)
                        stopping_detected = False

                        # Keep variable iter_ind[i] for each sample i
                        if self.iter_ind[sample_id] >= 1:
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
                            # Detect stopping token sequence and possibly interrupt gen
                            # for current sample
                            stopping_detected = detect_stop_tokens(
                                self.samples[sample_id], self.stop_tokens
                            )
                            # Update input pos (will be used in next pass)
                            self.input_pos[sample_id] = self.input_pos[sample_id][
                                -1:
                            ].add_(1)
                        else:
                            # First iteration for the current sample!
                            # Begin list of samples
                            self.samples[sample_id] = idx.view(1, -1)
                            # First iter for this sample, init KV cache!
                            self._init_sample_caches(sample_id, self.samples[sample_id])

                        # Send to next iff not at the last token
                        if (
                            self.iter_ind[sample_id] < self.max_new_tokens[sample_id]
                            and not stopping_detected
                        ):
                            # Only propagate last token (KV cache) - OR all initial
                            # prompt if 1st iter
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
                            print(
                                f"[DEBUG] Finished sample {sample_id}"
                                + f"{' - early detection' if stopping_detected else ''}"
                            )
                            # Release caches
                            self._delete_sample_caches(sample_id)

                            # Transmit msg with 'stop': true for this sample ID
                            out_msg = self._build_msg(
                                data="", sample_index=sample_id, stop=True
                            )
                            # Advertise the end of the generation for the sample
                            self.sample_finished[sample_id].set()

                        # Update iteration count for sample
                        self.iter_ind[sample_id] += 1

                        # NOTE: message queues will be the same if running in standalone
                        self.out_message_queue.append(out_msg)
                        self.out_queue_not_empty.set()

        if self.verb:
            print("[INFO] Stopping main thread (starter)")

    def secondary_loop(self) -> None:
        """
        Execution loop for non-starter nodes. This method must be used as the target of
        a thread that is launched once the node has been correctly initialized.

        The execution will be stopped once a PUT request is made to /stop.
        """
        assert self.model_config and self.model
        assert self.model_device is not None
        assert self.in_message_queue is not None and self.in_queue_not_empty
        assert self.out_message_queue is not None and self.out_queue_not_empty

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
        self.model.init_rope_mask(device=self.torch_device)
        self.model.eval()

        event_stop = threading.Event()
        loading_thread = threading.Thread(
            target=waiting_animation, args=("Processing samples", event_stop)
        )
        iter = 0

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
                        # Delete cached variables for sample id
                        self._delete_sample_caches(sample_id)

                        self.out_message_queue.append(in_msg)
                        self.out_queue_not_empty.set()
                    else:
                        idx = in_msg["data"].to(self.torch_device)
                        if sample_id not in self.T_i:
                            # Initialize caches - first pass for curr sample
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

        if self.verb:
            print("Node inference loop stopped")
