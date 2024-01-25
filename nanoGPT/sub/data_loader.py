#!/usr/bin/env python3

import os
from typing import Iterable, List, Tuple

import tiktoken  # TODO: add support for more complex tokenizers
import torch

from .char_tokenizer import CharacterTokenizer
from .model import GPTConfig


def load_dataset(
    input_path: str, tokenizer: CharacterTokenizer
) -> Tuple[torch.Tensor, CharacterTokenizer]:
    """
    Load a data set from a text file and tokenize its content.

    Args:
        input_path: path of the text file
        tokenizer (default: "char"): tokenizer to be used

    Returns:
        data: tensor containing the tokenized data
        tokenizer: the updated tokenizer (FIXME - see 24 below)
    """
    if not os.path.isfile(input_path):
        raise ValueError(f"Could not find {input_path}")

    fformats = (".txt", ".tex", ".md")
    if not input_path.lower().endswith(fformats):
        raise ValueError(
            f"File format not supported!\nSupported formats: {fformats}"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        f.close()

    if isinstance(tokenizer, CharacterTokenizer):
        # Initialize tokenizer
        tokenizer.tokenize(text)

        # Encode
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    else:
        raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")

    # FIXME: returning the updated tokenizer may not be necessary
    # (maybe try to use callbacks to update the internal state of the tokenizer)
    return data, tokenizer


def split_dataset(
    data: torch.Tensor, frac_train: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split the data set into training and validation set.

    Args:
        data: data set to be split
        frac_train: fraction of training elements
    """
    assert 0 <= frac_train <= 1
    n = int(frac_train * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return (train_data, val_data)


def get_batch(
    dataset: torch.Tensor,
    model_conf: GPTConfig,
):
    """
    Create batches (x - inputs and y - outputs) of contexts and targets.

    Args:
        dataset: the data set to be loaded to a tensor
        model_conf: the GPT configuration object
        device: the device on which to move the objects (default "cpu")

    Outputs:
        x: context inputs (each input is a block of size given by config)
        y: associated targets
    """
    # ix is used to randomize the order in the data set
    ix = torch.randint(
        len(dataset) - model_conf.block_size, (model_conf.batch_size,)
    )
    x = torch.stack([dataset[i : i + model_conf.block_size] for i in ix])
    # The "target" of a sequence is the next generated token immediately after
    # the considered block
    y = torch.stack(
        [dataset[i + 1 : i + model_conf.block_size + 1] for i in ix]
    )
    x, y = x.to(model_conf.device), y.to(model_conf.device)
    return x, y
