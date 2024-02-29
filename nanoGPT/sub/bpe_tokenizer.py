#!/usr/bin/env python3

import os
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import regex as re

"""
Implemetation of BPE tokenizer on byte encoding of UTF-8-coded characters.
"""

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
    by 'idx'.
    """
    newids = []
    i = 0
    while i < len(ids):
        # Jump to 1st occurrence of elem. in pair (prevent useless iterations)
        try:
            j = ids.index(pair[0], i)  # Fallback to i if not found
            newids.extend(ids[i:j])
            i = j
        except:
            # 1st element of pair not found -> copy all elem. in ids and stop
            newids.extend(ids[i:])
            break

        # If couple is present, replace, else add the element
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids


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

    merges = {}  # Mapping *byte* couples -> value of the merge
    vocab = {}  # Integers-to-bytes mapping (decoding)

    cache = {}  # Cache the translated words to make encoding faster

    def __init__(self):
        # RegEx for separating words and expressions (from Tiktoken, GPT-2)
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE,
        )

    def tokenize(self, text: str, out_vocab_size: int = 2000):
        """Train tokenizer from text"""
        # TODO: introduce word splitting at training
        # Convert text to list of 8-bit integers - byte representation [0,255]
        tokens = list(map(int, text.encode("utf-8")))

        # Number of iterations to get right output vocab size
        num_merges = out_vocab_size - 256
        ids = list(tokens)  # Copy original sequence (don't overwrite)

        for i in range(num_merges):
            stats = get_pairs_stats(ids)
            # Select pair with max occurrence (rank by the value)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i

            if VERB:
                print(f"Merging {top_pair} into new token {idx}")

            ids = replace_pair(ids, top_pair, idx)

            # We assume the new k-v pair is inserted in the LAST position (!)
            self.merges[top_pair] = idx

        compression_ratio = len(tokens) / len(ids)
        if VERB:
            print(f"Compression ratio: {compression_ratio:.2f}X")

        self.build_mapping()

    # TODO: save output files containing 'merges' and 'vocab'
    def store_tokenizer_info(self, info_dir: str):
        if not os.path.isdir(info_dir):
            raise FileNotFoundError(f"The directory {info_dir} does not exist!")

    def build_mapping(self):
        """
        Build the mapping from integer values to bytes/byte pairs.

        Can use this mapping to substitute the tokens (integers) with the full
        set of bytes they map to (even more than 2 if multiple pairs were
        linked)
        """
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # It's crucial this runs in the right order! NEED Python >= 3.7
        for (p0, p1), idx in self.merges.items():
            # Add to the vocab. the encoding of pairs
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]  # Concat. bytes
            # NOTE: this unpacks all couples into bytes! vocab[p] may already be
            # a couple!

    def encode(self, text: str) -> List[int]:
        if not self.trained():
            if VERB:
                print(
                    "Tokenizer was not initialized! Training tokenizer on provided text!"
                )
            self.tokenize(text)
        bpe_tokens = []
        # Isolate single "word" with regex
        for split in re.findall(self.pat, text):
            if split in self.cache:
                bpe_tokens.extend(self.cache[split])
            else:
                tokens = list(split.encode("utf-8"))
                finished_merges = False
                # Need at l. 2 tokens (else no pairs to merge and the function fails)
                while len(tokens) >= 2 and not finished_merges:
                    # Get set of "merge candidates"
                    stats = get_pairs_stats(tokens)
                    # This works because we assigned lower values to pairs we replaced first
                    # Use 'inf' as fallback for pairs that don't occur in self.merges
                    pair = min(
                        stats, key=lambda p: self.merges.get(p, float("inf"))
                    )
                    # Check whether the pair was supposed to be substituted
                    if pair not in self.merges:
                        finished_merges = True
                    else:
                        idx = self.merges[pair]
                        tokens = replace_pair(tokens, pair, idx)
                # Add the encoded word at the end of the list + cache
                self.cache[split] = tokens
                bpe_tokens.extend(tokens)

        return bpe_tokens

    def decode(self, ids: Iterable) -> str:
        """
        Given a list of integers (ids), return a python string.
        """
        assert len(self.vocab) > 0, "The tokenizer was not trained!"
        tokens = b"".join(self.vocab[idx] for idx in ids)
        # errors="replace" solves issues with utf-8 format
        text = tokens.decode("utf-8", errors="replace")
        return text

    def trained(self) -> bool:
        return len(self.vocab) > 0
