#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from sub import (DEVICE, EVAL_INTERVAL, EVAL_ITERS, LEARNING_RATE, N_EMBD,
                 N_HEADS, N_ITER_TRAIN, N_LAYER, CharacterTokenizer)
from sub.data_loader import get_batch, load_dataset, split_dataset
from sub.model import GPT, GPTConfig
from sub.utils import estimate_loss

VERB = True
CURR_DIR = os.path.dirname(__file__)


def main():
    in_file = os.path.join(CURR_DIR, "input.txt")
    # in_file = os.path.join(CURR_DIR, "divina_commedia.txt")

    tokenizer = CharacterTokenizer()
    data, tokenizer = load_dataset(in_file, tokenizer)  # data is a tensor
    vocab_size = tokenizer.vocab_size

    train_data, val_data = split_dataset(data, 0.9)

    # Dataloader
    torch.manual_seed(1337)

    model_config = GPTConfig()
    model_config.vocab_size = vocab_size

    xb, yb = get_batch(train_data, model_config)

    torch.manual_seed(1337)
    m = GPT(model_config)  # Model is already moved to device
    # Print number of parameters
    if VERB:
        print(f"Model runs on {model_config.device}")
        print(f"Number of parameters: {m.get_num_params() / 1e6} M")

    out, loss = m(xb, yb)
    if VERB:
        print("Output shape: ", out.shape)
        print(f"Initial loss: {loss}")

    # ---------------------- Training the model -------------------------------
    # Create PyTorch optimizer (AdamW)
    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    if VERB:
        print("------ Started training: ------ ")

    for iter in range(N_ITER_TRAIN):
        # Typical training loop
        # Every once in a while evaluate the loss (average for denoising)
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m, train_data, val_data)
            if VERB:
                print(
                    f"Step {iter}: "
                    f"training loss: {losses['train']:.4f}, "
                    f"validation loss {losses['val']:.4f}"
                )

        # Sample batch of data - NOTE: different at every iteration (get_batch is random)
        xb, yb = get_batch(train_data, model_config)

        # Evaluate loss
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if VERB:
        print(f"Loss: {loss.item()}")

    # Switch to eval mode and generate text
    m.eval()
    # Start generation by feeding tensor [[0]]
    idx = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    gen_text = tokenizer.decode(
        m.generate(idx, max_new_tokens=10000)[0].tolist()
    )
    if VERB:
        print("After training:")
        print(gen_text)


if __name__ == "__main__":
    main()
