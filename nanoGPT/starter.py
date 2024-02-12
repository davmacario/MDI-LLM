#!/usr/bin/env python3

import logging
import os

import cherrypy as cp
import torch

from sub.model_dist import GPTDistributed

# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
log_file = os.path.join(script_dir, "logs", "logs_starter.log")
if not os.path.exists(os.path.dirname(log_file)):
    os.mkdir(os.path.dirname(log_file))
log_wp = logging.getLogger("model_dist")
hdlr = logging.StreamHandler()
fhdlr = logging.FileHandler(log_file, mode="w")
formatter = logging.Formatter("%(name)s → %(levelname)s: %(message)s")
fhdlr.setFormatter(formatter)
hdlr.setFormatter(formatter)
log_wp.addHandler(hdlr)
log_wp.addHandler(fhdlr)
log_wp.setLevel(logging.DEBUG)

dataset = "shakespeare"

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

if __name__ == "__main__":
    dataset_name = os.path.splitext(dataset)[0]
    data_dir = os.path.join(script_dir, "data", dataset_name)
    out_dir = os.path.join(data_dir, "out")
    settings_path = os.path.join(script_dir, "settings_distr")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    network_conf_path = os.path.join(settings_path, "configuration.json")

    gpt_distr = GPTDistributed(ckpt_path)

    # Operation
    try:
        gpt_distr.start()
    except KeyboardInterrupt:
        cp.engine.stop()
        print("Starter stopped")
