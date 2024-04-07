import os
import sys
from typing import Any, Dict, List, Mapping

import tiktoken
import torch

from sub.bpe_tokenizer import BPETokenizer
from sub.char_tokenizer import CharacterTokenizer
from sub.config import (DEVICE, DTYPE, HEADERLENGTH, N_LAYERS_NODES, PLOTS,
                        TEMPERATURE, TOP_K, VERB)
from sub.model import GPT, GPTConfig
from sub.utils import (count_model_layers, find_eot, get_prompt, load_from_hf,
                       loading_bar, plot_tokens_per_time, split_parameters)


class ModelParallelGPT(GPT):
    """
    Model-Parallel implementation of sub.model.GPT (GPT-2).
    This allows to train the model on multiple GPUs of a single host.

    Notice that this implementation, even though it can be used to perform inference,
    does not allow for pipelining (as Model-Distributed Inference does).
    This class was specifically though for model-parallel training (as opposed to the
    use of Torch's DistributedDataParallel) for systems where the full model (+
    backpropagation intermediate results) cannot fit on a single GPU, but may fit in
    multiple GPUs.

    This code only allows to use multiple CUDA-enabled GPUs (and additionally CPU to
    allow for offloading).
    """

    def __init__(self, config: GPTConfig, devices: List[str], *args, **kwargs):
        super().__init__(config, **kwargs)

        # FIXME: for now I did not write the "partition maps" for more than 3 devices/nodes
        assert len(devices) <= 3, "Max number of devices supported is 3"

        n_detected_dev = torch.cuda.device_count()
        for dev in devices:
            if "cuda" not in dev and "cpu" not in dev:
                raise RuntimeError(f"Unsupported device: {dev}")
            if "cuda" in dev and dev[-1].isnumeric():
                dev_index = int(dev.split(":")[-1])
                if dev_index >= n_detected_dev:
                    raise RuntimeError(
                        f"Invalid device index: {dev_index} - {n_detected_dev} detected"
                    )

        self.devices = devices

        # TODO: Split the model layers - maybe use same map as with MDI???
        # OCD kicking in (plus, CPU gets placed before all "cuda:x")
        devices.sort()

        n_layers = config.n_layer
        self.partition_map = N_LAYERS_NODES[len(devices)][n_layers]

        self.model_layers = []
        self.split_layers()

    def split_layers(self):
        """
        Initialize the model layers on the devices
        """
        assert self.partition_map is not None, "The layer partition map was not set!"

        # TODO
        # Idea: the layers are grouped in a list

    @classmethod
    def from_pretrained(
        cls, model_type: str, devices: List[str], override_args=None, **kwargs
    ):
        """
        Load a model from a checkpoint/pretrained model.
        NOTE: this method for the GPT class only allows to load parameters from GPT2
        flavors, while here it is not possible to trivially copy the state dict...

        Args:
            model_name: can either be a .pt file found on disk, or a GPT-2 flavor.
            devices: list of devices to be used to parallelize the model.
        """
        # TODO
        pass

    def forward(self, idx, targets):
        # TODO
        pass

    def dump_parameters(self, out_file=None):
        """
        Load the parameters to a file/buffer so that the state_dict is compatible with
        the original GPT class.

        This will require some key-fixing magic and a lot of time :).

        Don't only dump to file, but allow to return the parameter as variable (e.g. if
        out_file is None) to include the state dict in the ckpt
        """
        # TODO
        pass
