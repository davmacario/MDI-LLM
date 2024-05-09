#!/usr/bin/env python3

import os
from pathlib import Path

import torch
from sub import Config
from sub.utils import count_transformer_blocks, split_parameters

"""
Use this script to:
- Download weights from Huggingface Hub
- Store them in a local folder
- Partition them among a number of nodes
- Store the partitions at a specific location
"""
