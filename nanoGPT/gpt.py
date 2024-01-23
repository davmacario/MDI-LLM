#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from sub.config import (BATCH_SIZE, BLOCK_SIZE, DEVICE, DROPOUT, EVAL_INTERVAL,
                        EVAL_ITERS, LEARNING_RATE, N_EMBD, N_HEADS,
                        N_ITER_TRAIN, N_LAYER)

VERB = True
CURR_DIR = os.path.dirname(__file__)


class Head(nn.Module):
    """Single self-attention head"""

    # See thesis-notes.md for detail explaination

    def __init__(self, head_size: int):
        super().__init__()

        # Linear projection
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # 'tril' is not a parameter - assign it to a buffer
        self.register_buffer(
            "tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor):
        """Forward pass, single attention head"""
        _, T, _ = x.shape  # (B, T, C)
        k = self.key(x)  # (B, T, hs) - hs: "head size"
        q = self.query(x)  # (B, T, hs)

        # Compute attention scores
        # Scaled self-attention
        hs = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (hs**-0.5)  # (B, T, hs) @ (B, hs, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Weighted aggregation
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module: use multiple attention heads in parallel.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        n_embd = num_heads * head_size

        # Create a Module List containing all the heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection - linear transform. of the output of attention heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # The output is the concatenation of the outputs from each head
        # Concat. on "last" dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Simple linear layer, used after MHA"""

    def __init__(self, n_embd):
        """
        Create feed-forward layer.

        Args:
            n_embd: number of input and output embeddings (i.e., neurons)
        """
        super().__init__()
        # The 4* comes from the Transformer paper + the final linear layer is
        # necessary for residual connections & it brings the dimension down
        # to the value we want (n_embd)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Projection layer
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block - communication + computation

    Note: decoder only
    """

    def __init__(self, n_embd, n_head):
        """
        Instantiate Transformer block

        Args:
            n_embd: number of token embeddings per time batch
            n_head: number of attention heads (must be a divisor of n_embd, as
                each head will work with dim. n_embd // n_head)
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # LayerNorm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor):
        # NOTE: using residual connections (sum the inputs to the outputs)
        # LayerNorm applied before each layer
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """Bigram language model using MHA"""

    def __init__(self, vocab_size):
        """
        Create BigramLanguageModel object.

        Args:
            vocab_size: size of the used vocabulary, necessary for token
                embedding
        """
        super().__init__()
        # Each token will read the logits for the next token from a lookup table
        # Embedding table of size vocab_size x vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        # Positional embedding - not useful now, bigram is translation-invariant
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEADS) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Forward pass in bigram embedding.

        Args:
            idx: (Batch size) x (Time) tensor of integers
            targets: target embedding, same size as idx

        Returns:
            logits - row 'idx' of the token embedding table; size is
                BxTxvocab_size
        """
        B, T = idx.shape

        # The logits returned are the ones in row idx of the table
        # This is arranged in a tensor of size Batch x Time x Channel(=N_EMBED)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            # Conform to PyTorch's specs
            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Generate new tokens using the Bigram Language Model, provided the input
        sequence of integers "idx".

        Args:
            idx: input sequence of encoded chars/words (B x T)
            max_new_tokens: maximum number of tokens to be generated

        Returns:
            Updated version of idx, containing the generated elements
        """
        for _ in range(max_new_tokens):
            # NOTE: crop idx to the last 'block_size' tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus only on last time step (dim: (B, C))
            # Now, the model chooses the next token based on the last one only
            logits = logits[:, -1, :]  # B x C
            # Apply softmax to get probabilities of tokens in the last time step
            probs = F.softmax(logits, dim=1)
            # Sample p.m.f
            idx_next = torch.multinomial(probs, num_samples=1)  # B x 1
            # Append sampled index to idx
            idx = torch.cat((idx, idx_next), dim=1)  # B x (T+1)

        return idx


def main():
    in_file = os.path.join(CURR_DIR, "input.txt")

    with open(in_file, "r", encoding="utf-8") as f:
        text = f.read()
        f.close()

    # Create dictionary (characters)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}  # String to integer
    encode = lambda string: [stoi[c] for c in string]

    itos = {i: ch for i, ch in enumerate(chars)}  # Integer to string
    decode = lambda line: "".join([itos[i] for i in line])

    # Encode and move to tensor
    data = torch.tensor(encode(text), dtype=torch.long)

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
    model = BigramLanguageModel(vocab_size)
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

    idx = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    gen_text = decode(m.generate(idx, max_new_tokens=10000)[0].tolist())
    if VERB:
        print("After training:")
        print(gen_text)


if __name__ == "__main__":
    main()
