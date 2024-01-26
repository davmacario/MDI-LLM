#!/usr/bin/env python3

from typing import Iterable, List, Union


class CharacterTokenizer:
    """
    Character-based tokenizer.

    Each character is encoded with a specific token.
    """

    init = False

    def tokenize(self, text: Union[str, Iterable]):
        # Create dictionary (characters)
        self.dictionary = sorted(list(set(text)))
        self.n_vocab = len(self.dictionary)

        # String to integer mapping
        self.stoi = {ch: i for i, ch in enumerate(self.dictionary)}
        # Integer to string mapping
        self.itos = {i: ch for i, ch in enumerate(self.dictionary)}

        self.init = True

    def encode(self, in_str: str) -> List:
        if not self.init:
            # "Train" tokenizer on passed sequence
            self.tokenize(in_str)
        return [self.stoi[c] for c in in_str]

    def decode(self, line: Iterable) -> str:
        assert (
            self.init
        ), "Tokenizer was not initialized - this tokenizer infers the dictionary from the text"
        return "".join([self.itos[i] for i in line])
