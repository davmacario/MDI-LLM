#!/usr/bin/env python3

import json
import os

import cherrypy as cp
import torch

from sub.model_dist import GPTServer

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    script_dir = os.path.dirname(__file__)
    settings_path = os.path.join(script_dir, "settings_distr")
    network_conf_path = os.path.join(settings_path, "configuration.json")

    try:
        with open(network_conf_path, "r") as f:
            full_config = json.load(f)
            gpt_webserv = GPTServer(
                node_config=full_config["nodes"]["intermediate"][0]
            )
    except KeyboardInterrupt:
        print("Node stopped!")
        cp.engine.stop()
