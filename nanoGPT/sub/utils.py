#!/usr/bin/env python3

import torch

from .model import GPTConfig


def get_batch(split: str):
    """
    Create batches (x - inputs and y - outputs) of contexts and targets.

    Args:
        split: string, can be "train" of "val"

    Outputs:
        x: context inputs
        y: associated targets
    """
    assert split in {"train", "val"}

    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()  # Tell the program not to evaluate the gradients (no BP)
def estimate_loss():
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
    # Set model to evaluation mode
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Re-set the model to training mode
    model.train()
    return out


xb, yb = get_batch("train")
