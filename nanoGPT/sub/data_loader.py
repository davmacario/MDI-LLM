#!/usr/bin/env python3

import os
from typing import Iterable

import tiktoken  # TODO: add support for more complex tokenizers

from .char_tokenizer import CharacterTokenizer


def load_dataset(input_path: str, tokenizer: CharacterTokenizer):
    """
    Load a data set from a text file and tokenize its content.

    Args:
        input_path: path of the text file
        tokenizer (default: "char"): tokenizer to be used

    Returns:
        data: tokenized data

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
        data = tokenizer.encode(text)
    else:
        raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")

    return data, tokenizer


def split_dataset(data: Iterable, frac_train: float = 0.9):
    """
    Split the data set into training and validation set
    """
    pass
