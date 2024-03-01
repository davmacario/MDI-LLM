#!/usr/bin/env python3

import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import regex as re

"""
Implemetation of BPE tokenizer on byte encoding of UTF-8-coded characters.
"""

VERB = True


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, "big")


def bytes_to_int(xb: bytes) -> int:
    return int.from_bytes(xb, "big")


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

    Public API:
        tokenize: train tokenizer
        encode: encode a sequence (if not already trained, will
        decode
    """

    merges = {}  # Mapping *byte* couples -> value of the merge
    vocab = {}  # Integers-to-bytes mapping (decoding) - Actually, int -> int
    n_vocab = 256

    cache = {}  # Cache the translated words to make encoding faster

    def __init__(
        self,
        voc_path: Union[None, str] = None,
        merges_path: Union[None, str] = None,
    ):
        # RegEx for separating words and expressions (from Tiktoken, GPT-2)
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE,
        )

        if voc_path is not None and merges_path is not None:
            self.load_tokenizer_info(voc_path, merges_path)

    def tokenize(self, text: str, out_vocab_size: int = 500):
        """Train tokenizer from text"""
        self.n_vocab = out_vocab_size
        # FIXME: need to handle case when there is no more tokens to merge
        tokens = []
        for split in re.findall(self.pat, text):
            # Convert text to list of 8-bit integers - byte representation [0,255]
            tokens.append(list(map(int, split.encode("utf-8"))))

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

        len_init = sum([len(x) for x in tokens])
        len_fin = sum([len(x) for x in ids])
        compression_ratio = len_init / len_fin
        if VERB:
            print(f"Compression ratio: {compression_ratio:.2f}X")

        self.build_mapping()

    def build_mapping(self):
        """
        Build the mapping from integer values to bytes/byte pairs.

        Can use this mapping to substitute the tokens (integers) with the full
        set of bytes they map to (even more than 2 if multiple pairs were
        linked)
        """
        tmp_vocab = {idx: bytes([idx]) for idx in range(256)}
        # It's crucial this runs in the right order! NEED Python >= 3.7
        for (p0, p1), idx in self.merges.items():
            # Add to the vocab. the encoding of pairs
            tmp_vocab[idx] = tmp_vocab[p0] + tmp_vocab[p1]  # Concat. bytes
            # NOTE: this unpacks all couples into bytes! vocab[p] may already be
            # a couple!

        for idx in tmp_vocab:
            self.vocab[idx] = bytes_to_int(tmp_vocab[idx])

    def store_tokenizer_info(self, info_dir: str, overwrite=False):
        """
        Store the tokenizer information to output files "merges.bpe" and
        "encoder.json", placed in the given directory.

        If overwrite = True, it will overwrite existing files.
        """
        if not (os.path.exists(info_dir) and os.path.isdir(info_dir)):
            raise FileNotFoundError(f"The directory {info_dir} does not exist!")

        # Save merges as .bpe file (txt)
        merges_path = os.path.join(info_dir, "merges.bpe")
        if os.path.exists(merges_path):
            if overwrite:
                warnings.warn("Overwriting merges file!")
            else:
                warnings.warn("Merges file already exists!")
                return
        with open(merges_path, "w", encoding="utf-8") as f:
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")
            f.close()

        # Save "inverted" vocab as json
        vocab_path = os.path.join(info_dir, "encoder.json")
        if os.path.exists(vocab_path):
            if overwrite:
                warnings.warn("Overwriting vocabulary file!")
            else:
                warnings.warn("Vocabulary file already exists!")
                return
        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f)
            f.close()

    def load_tokenizer_info(self, vocab_path: str, meta_path: str):
        """
        Initialize tokenizer mappings and vocabulary from files.

        The provided directory must contain a 'merges.bpe' and an 'encoder.json'
        """
        for pt in [vocab_path, meta_path]:
            if not os.path.exists(pt):
                raise FileNotFoundError(f"Invalid path {pt}")

        if self.merges != {} or self.vocab != {}:
            raise ValueError("Tokenizer is already initialized!")

        # Load merges
        with open(meta_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()
            f.close()

        self.merges = {
            (int(p[0]), int(p[1])): int(p[2])
            for line in bpe_data.split("\n")
            for p in line.split()
        }

        # Load vocab
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
            f.close()
        self.n_vocab = len(self.vocab)

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
        tokens = b"".join([int_to_bytes(self.vocab[idx]) for idx in ids])
        # errors="replace" solves issues with utf-8 format
        text = tokens.decode("utf-8", errors="replace")
        return text

    def trained(self) -> bool:
        """Returns true iff the tokenizer has already been trained"""
        self.n_vocab = len(self.vocab)
        return self.n_vocab > 0
