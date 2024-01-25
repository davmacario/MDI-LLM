#!/usr/bin/env python3

import torch

from .config import EVAL_ITERS  # N. of elements over which loss is averaged
from .data_loader import get_batch
from .model import GPT


@torch.no_grad()  # Tell the program not to evaluate the gradients (no BP)
def estimate_loss(
    model: GPT,
    train: torch.Tensor,
    val: torch.Tensor,
):
    """
    Evaluate the mean loss over a fixed number of iterations during training.
    This allows to remove possible noise and provide more meaningful
    results.

    Returns:
        Dict containing the keys:
            "train": mean loss over EVAL_ITERS iterations for training set
            "val": mean loss over EVAL_ITERS iterations for validation set
    """
    out = {}
    dss = {
        "train": train,
        "val": val,
    }
    # Set model to evaluation mode
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(dss[split], model.config)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Re-set the model to training mode
    model.train()
    return out
