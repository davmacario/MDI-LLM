#!/usr/bin/env python3

from typing import Iterable, List


class CharacterTokenizer:
    """
    Character-based tokenizer.

    Each character is encoded with a specific token.
    """

    init = False

    def tokenize(self, text: str | Iterable):
        # Create dictionary (characters)
        self.dictionary = sorted(list(set(text)))
        self.vocab_size = len(self.dictionary)

        # String to integer mapping
        self.stoi = {ch: i for i, ch in enumerate(self.dictionary)}
        # Integer to string mapping
        self.itos = {i: ch for i, ch in enumerate(self.dictionary)}

        self.init = True

    def encode(self, in_str: str) -> List:
        assert (
            self.init
        ), "Tokenizer was not initialized - this tokenizer infers thedictionary from the text"
        return [self.stoi[c] for c in in_str]

    def decode(self, line: Iterable) -> str:
        assert (
            self.init
        ), "Tokenizer was not initialized - this tokenizer infers the dictionary from the text"
        return "".join([self.itos[i] for i in line])
