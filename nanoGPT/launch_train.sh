#!/bin/bash

python3 ./nanoGPT/train.py --batch-size 24 --verb --init scratch --ckpt-interval 500
