#!/usr/bin/env python3

import json
import os
from contextlib import nullcontext

import cherrypy as cp
import torch

from sub.config import DEVICE, DTYPE
from sub.model_dist import GPTServer

# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
settings_path = os.path.join(script_dir, "settings_distr")
network_conf_path = os.path.join(settings_path, "configuration.json")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# for later use in torch.autocast:
if "cuda" in DEVICE:
    device_type = "cuda"
elif "mps" in DEVICE:
    device_type = "mps"
else:
    device_type = "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[DTYPE]
ctx = (
    nullcontext()
    if device_type in {"cpu", "mps"}
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

try:
    with open(network_conf_path, "r") as f:
        full_config = json.load(f)
        gpt_webserv = GPTServer(
            node_config=full_config["nodes"]["intermediate"][0]
        )
except KeyboardInterrupt:
    print("Node stopped!")
    cp.engine.stop()
