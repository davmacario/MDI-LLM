#!/usr/bin/env python3

from typing import Dict, Iterable, List, Mapping, Tuple, Union

VERB = True


def get_pairs_stats(ids: List[int]) -> Mapping[Tuple[int], int]:
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


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer (from A. Karpathy)

    To be implemented (common API):
        Methods:
            encode
            decode
    """

    merges = {}

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
            self.merges[top_pair] = idx

        compression_ratio = len(tokens) / len(ids)
        if VERB:
            print(f"Compression ratio: {compression_ratio:.2f}X")
