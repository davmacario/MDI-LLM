#!/usr/bin/env python3

import os
from typing import Iterable, List, Tuple, Union

import numpy as np
import tiktoken
import torch

from .bpe_tokenizer import BPETokenizer
from .char_tokenizer import CharacterTokenizer
from .model import GPTConfig


def load_dataset(
    input_path: str,
    tokenizer: Union[CharacterTokenizer, BPETokenizer, tiktoken.Encoding],
) -> List:
    """
    Load a data set from a text file and tokenize its content.

    Args:
        input_path: path of the text file
        tokenizer: tokenizer to be used (can be char-based or from tiktoken)

    Returns:
        data: List containing the tokens
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
        # Encode and move to tensor
        # NOTE: the tokenizer gets updated automatically
        data = tokenizer.encode(text)
    elif isinstance(tokenizer, BPETokenizer):
        assert tokenizer.trained(), "Tokenizer was not trained!"
        data = tokenizer.encode(text)
    elif isinstance(tokenizer, tiktoken.Encoding):
        data = tokenizer.encode_ordinary(text)
    else:
        raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")

    return data


def split_dataset(
    data: torch.Tensor, frac_train: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split the data set into training and validation set.

    NOTE: data is split in-place - i.e., not shuffled (should not lose
    relationship between consecutive tokens)

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
    dataset,
    model_conf: GPTConfig,
):
    """
    Create batches (x - inputs and y - outputs) of contexts and targets.

    Args:
        dataset: the data set to be loaded to a tensor (tensor/np array/...)
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
    if type(dataset) == torch.Tensor:
        x = torch.stack([dataset[i : i + model_conf.block_size] for i in ix])
        # The "target" of a sequence is the next generated token immediately after
        # the considered block
        y = torch.stack(
            [dataset[i + 1 : i + model_conf.block_size + 1] for i in ix]
        )
    elif type(dataset) in {np.memmap or np.ndarray}:
        x = torch.stack(
            [
                torch.from_numpy(
                    (dataset[i : i + model_conf.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (dataset[i + 1 : i + 1 + model_conf.block_size]).astype(
                        np.int64
                    )
                )
                for i in ix
            ]
        )
    else:
        raise TypeError(f"Unsuppoerted data type {type(dataset)}")

    if model_conf.device == "cuda":
        x, y = x.pin_memory().to(
            model_conf.device, non_blocking=True
        ), y.pin_memory().to(model_conf.device, non_blocking=True)
    else:
        x, y = x.to(model_conf.device), y.to(model_conf.device)
    return x, y
