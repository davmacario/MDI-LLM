#!/usr/bin/env bash

./train.py -v --ckpt checkpoints/custom/NanoLlama/ --dataset data/openwebtext/ --batch-size 2 --au --grad-acc-steps 20
