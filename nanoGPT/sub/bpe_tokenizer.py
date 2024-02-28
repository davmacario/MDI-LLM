#!/usr/bin/env python3

from typing import Dict, Iterable, List, Mapping, Tuple, Union

import regex as re

VERB = True


def get_pairs_stats(ids: List[int]) -> Mapping[Tuple[int, int], int]:
    """
    Considering the couples of subsequent elements in the input list, return
    the occurrence of each pair.

    Args:
        ids: list of integer values (order matters)

    Returns:
        mapping between each pair and the n. of occurrences
    """
    counts = {}

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def replace_pair(ids: List[int], pair: Tuple[int, int], idx: int):
    """
    Return copy of 'ids' where the consecutive pairs 'pair' have been replaced
    by 'idx'
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids


# TODO: save output files containing 'merges' and 'vocab'
# TODO: add regex - need to figure out training
# TODO: include new tokenizer in model


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer (from A. Karpathy)

    To be implemented (common API):
        Methods:
            encode
            decode
    """

    merges = {}
    vocab = {}  # Integers-to-bytes mapping (decoding)

    def __init__(self):
        pass

    def tokenize(self, text: str, out_vocab_size: int = 512):
        """Create tokenizer from text"""
        # Convert text to list of integers (8-bit)
        tokens = list(map(int, text.encode("utf-8")))

        num_merges = out_vocab_size - 256
        ids = list(tokens)  # Copy original sequence

        for i in range(num_merges):
            stats = get_pairs_stats(ids)
            # Select pair with max occurrence (rank by the value)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i

            if VERB:
                print(f"Merging {top_pair} into new token {idx}")

            ids = replace_pair(ids, top_pair, idx)

            # We assume the new k-v pair is inserted in the LAST position
            self.merges[top_pair] = idx

        compression_ratio = len(tokens) / len(ids)
        if VERB:
            print(f"Compression ratio: {compression_ratio:.2f}X")

        self.build_mapping()

    def build_mapping(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # It's crucial this runs in the right order! NEED Python >= 3.7
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]  # Concat. 2 bytes

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8"))
        finished_merges = False
        # Need at l. 2 tokens (else no pairs to merge and the function fails)
        while len(tokens) >= 2 and not finished_merges:
            # Get set of "merge candidates"
            stats = get_pairs_stats(tokens)
            # This works because we assigned lower values to pairs we replaced first
            # Use 'inf' as fallback for pairs that don't occur in self.merges
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # ISSUE: this fails if no pairs can be merged
            if pair not in self.merges:
                finished_merges = True
            else:
                idx = self.merges[pair]
                tokens = replace_pair(tokens, pair, idx)

        return tokens

    def decode(self, ids: Iterable) -> str:
        """
        Given a list of integers (ids), return a python string.
        """
        assert len(self.vocab) > 0, "The tokenizer was not trained!"
        tokens = b"".join(self.vocab[idx] for idx in ids)
        # Issue: may have decoding issues - UTF-8 requires specific byte format
        # -> Add errors="replace"
        text = tokens.decode("utf-8", errors="replace")
        return text
