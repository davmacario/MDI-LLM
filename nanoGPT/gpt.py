#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from sub import (BATCH_SIZE, BLOCK_SIZE, DEVICE, DROPOUT, EVAL_INTERVAL,
                 EVAL_ITERS, LEARNING_RATE, N_EMBD, N_HEADS, N_ITER_TRAIN,
                 N_LAYER, CharacterTokenizer, load_dataset)

VERB = True
CURR_DIR = os.path.dirname(__file__)


def main():
    in_file = os.path.join(CURR_DIR, "input.txt")
    # in_file = os.path.join(CURR_DIR, "divina_commedia.txt")

    tokenizer = CharacterTokenizer()
    data, tokenizer = load_dataset(in_file, tokenizer)
    vocab_size = tokenizer.vocab_size

    # Separate in train and validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Dataloader
    torch.manual_seed(1337)

    def get_batch(split: str):
        """
        Create batches (x - inputs and y - outputs) of contexts and targets.

        Args:
            split: string, can be "train" of "val"

        Outputs:
            x: context inputs
            y: associated targets
        """
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

    torch.manual_seed(1337)
    model = GPT(vocab_size)
    m = model.to(DEVICE)
    # Print number of parameters
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    out, loss = m(xb, yb)
    if VERB:
        print(out.shape)
        print(f"Loss: {loss}")

    # --------------- Training the Bigram model -----------------------
    # Create PyTorch optimizer (AdamW)
    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    if VERB:
        print("Started training:")

    for iter in range(N_ITER_TRAIN):
        # Typical training loop

        # Every once in a while eval. the loss (denoise)
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            if VERB:
                print(
                    f"Step {iter}: training loss: {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

        # Sample batch of data
        xb, yb = get_batch("train")

        # Evaluate loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if VERB:
        print(f"Loss: {loss.item()}")

    # Switch to eval mode and generate text
    m.eval()
    # Start generation by feeding tensor [[0]]
    idx = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    gen_text = decode(m.generate(idx, max_new_tokens=10000)[0].tolist())
    if VERB:
        print("After training:")
        print(gen_text)


if __name__ == "__main__":
    main()
