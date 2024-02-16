#!/usr/bin/env python3

import json
import logging
import os

import cherrypy as cp
import torch

from sub.model_dist import GPTServer

# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
log_file = os.path.join(script_dir, "logs", "logs_finisher.log")
if not os.path.exists(os.path.dirname(log_file)):
    os.mkdir(os.path.dirname(log_file))
log_wp = logging.getLogger("model_dist")
fhdlr = logging.FileHandler(log_file, mode="w")
formatter = logging.Formatter("[%(asctime)s] → %(levelname)s: %(message)s")
fhdlr.setFormatter(formatter)
log_wp.addHandler(fhdlr)
log_wp.setLevel(logging.DEBUG)

if __name__ == "__main__":
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    settings_path = os.path.join(script_dir, "settings_distr")
    network_conf_path = os.path.join(settings_path, "configuration.json")

    try:
        with open(network_conf_path, "r") as f:
            full_config = json.load(f)
            gpt_webserv = GPTServer(
                node_config=full_config["nodes"]["finisher"]
            )
    except KeyboardInterrupt:
        print("Node stopped!")
        cp.engine.stop()
