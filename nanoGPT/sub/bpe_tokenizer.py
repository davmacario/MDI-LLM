#!/usr/bin/env python3

import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import regex as re

"""
Implemetation of BPE tokenizer on byte encoding of UTF-8-coded characters.
"""

VERB = True


def get_pairs_stats(ids: List[Any]) -> Mapping[Tuple[int, int], int]:
    """
    Considering the couples of subsequent elements in the input list, return
    the occurrence of each pair.

    Args:
        ids: list of integer values (order matters) or list of list of ints (if
            using regex to separate words)

    Returns:
        mapping between each pair and the n. of occurrences
    """
    counts = {}

    if isinstance(ids[0], int):
        lst = [ids]
    elif isinstance(ids[0], list):
        lst = ids
    else:
        raise ValueError(f"Unsupported type for argument: {type(ids)}")

    for sublist in lst:
        for pair in zip(sublist, sublist[1:]):
            counts[pair] = counts.get(pair, 0) + 1

    return counts


def replace_pair(
    ids: List[Any], pair: Tuple[int, int], idx: int
) -> Union[List[int], List[List[int]]]:
    """
    Return copy of 'ids' where the consecutive pairs 'pair' have been replaced
    by 'idx'.

    Note: the variable 'ids' can also be a list of lists containing in each
    sublist the tokens of separate words (from regex).
    """
    if isinstance(ids[0], int):
        lst = [ids]
    elif isinstance(ids[0], list):
        lst = ids
    else:
        raise ValueError(f"Unsupported type for argument: {type(ids)}")

    newids = []
    for sublist in lst:
        i = 0
        newids_word = []
        while i < len(sublist):
            # Jump to 1st occurrence of elem. in pair (prevent useless iterations)
            try:
                j = sublist.index(pair[0], i)  # Fallback to i if not found
                newids_word.extend(sublist[i:j])
                i = j
            except:
                # 1st element of pair not found -> copy all elem. in ids and stop
                newids_word.extend(sublist[i:])
                break

            # If couple is present, replace, else add the element
            if (
                i < len(sublist) - 1
                and sublist[i] == pair[0]
                and sublist[i + 1] == pair[1]
            ):
                newids_word.append(idx)
                i += 2
            else:
                newids_word.append(sublist[i])
                i += 1
        newids.append(newids_word)

    if len(newids) == 1:
        return newids[0]
    else:
        return newids


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
    n_vocab = 256

    cache = {}  # Cache the translated words to make encoding faster

    def __init__(self):
        # RegEx for separating words and expressions (from Tiktoken, GPT-2)
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE,
        )

    def tokenize(self, text: str, out_vocab_size: int = 1000):
        """Train tokenizer from text"""
        self.n_vocab = out_vocab_size
        if self.pat is not None:
            # FIXME: need to handle case when there is no more tokens to merge
            tokens = []
            for split in re.findall(self.pat, text):
                # Convert text to list of 8-bit integers - byte representation [0,255]
                tokens.append(list(map(int, split.encode("utf-8"))))
        else:
            # FIXME: not consistent...
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
        if not (os.path.exists(info_dir) and os.path.isdir(info_dir)):
            raise FileNotFoundError(f"The directory {info_dir} does not exist!")

        # Save merges as .bpe file (txt)
        with open(
            os.path.join(info_dir, "merges.bpe"), "w", encoding="utf-8"
        ) as f:
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")
            f.close()

        # Save "inverted" vocab as json
        inv_vocab = {v: k for k, v in self.vocab.items()}
        with open(os.path.join(info_dir, "encoder.json"), "w") as f:
            json.dump(inv_vocab, f)
            f.close()

    def load_data(self, info_dir: str):
        if not (os.path.exists(info_dir) and os.path.isdir(info_dir)):
            raise FileNotFoundError(f"The directory {info_dir} does not exist!")

        if self.merges != {} or self.vocab != {}:
            raise ValueError("Tokenizer is already initialized!")

        # Load merges
        with open(
            os.path.join(info_dir, "merges.bpe"), "r", encoding="utf-8"
        ) as f:
            bpe_data = f.read()
            f.close()

        self.merges = {
            (p[0], p[1]): p[2]
            for line in bpe_data.split("\n")
            for p in line.split()
        }

        # Load vocab
        with open(os.path.join(info_dir, "encoder.json"), "r") as f:
            inv_vocab = json.load(f)
            self.vocab = {v: k for k, v in inv_vocab.items()}
            f.close()

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
