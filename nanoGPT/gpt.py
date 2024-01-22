#!/usr/bin/env

import os

import torch
import torch.nn as nn
from torch.nn import functional as F

VERB = True
CURR_DIR = os.path.dirname(__file__)
BLOCK_SIZE = 8  # (context length)
BATCH_SIZE = 4


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Each token will read the logits for the next token from a lookup table
        # Embedding table of size vocab_size x vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        """Forward pass in bigram embedding"""
        # The logits returned are the ones in row idx of the table
        # This is arranged in a tensor of size Batch x Time x Channel
        logits = self.token_embedding_table(idx)
        return logits


def main():
    in_file = os.path.join(CURR_DIR, "input.txt")

    with open(in_file, "r", encoding="utf-8") as f:
        text = f.read()
        f.close()

    if VERB:
        print(f"Length of the input file in characters: {len(text)}")
        print(text[:1000])

    # Create dictionary (characters)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    if VERB:
        print(f"Vocabulary:\n{''.join(chars)}")
        print(f"Vocabulary size: {vocab_size}")

    # Tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}  # String to integer
    encode = lambda string: [stoi[c] for c in string]

    itos = {i: ch for i, ch in enumerate(chars)}  # Integer to string
    decode = lambda line: "".join([itos[i] for i in line])

    # Encode and move to tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    if VERB:
        print(data.shape, data.dtype)
        print(data[:1000])

    # Separate in train and validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    if VERB:
        # Observe context/block - this is what the transformer learns!
        x = train_data[:BLOCK_SIZE]
        y = train_data[1 : BLOCK_SIZE + 1]
        for t in range(BLOCK_SIZE):
            context = x[: t + 1]
            target = y[t]
            print(f"When the input is {context}, the target is {target}")

    # Training
    torch.manual_seed(1337)

    def get_batch(split: str):
        """
        Create batches (x - inputs and y - outputs) of contexts and targets.

        Args:
            split: string, can be "train" of "validation"

        Outputs:
            x: context inputs
            y: associated targets
        """
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i * 1 : i * BLOCK_SIZE + 1] for i in ix])
        return x, y

    xb, yb = get_batch("train")
    if VERB:
        print("Inputs:")
        print(xb.shape)
        print(xb)
        print("Targets:")
        print(yb.shape)
        print(yb)

        print("-----")

        for b in range(BATCH_SIZE):
            for t in range(BLOCK_SIZE):
                context = xb[b, : t + 1]
                target = yb[b, t]
                print(
                    f"When input is {context.tolist()}, the target is {target}"
                )

    torch.manual_seed(1337)
    m = BigramLanguageModel(vocab_size)
    out = m(xb, yb)
    if VERB:
        print(out.shape)


if __name__ == "__main__":
    main()
