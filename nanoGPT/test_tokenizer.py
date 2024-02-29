#!/usr/bin/env python3

import os

from sub import BPETokenizer

script_dir = os.path.dirname(__file__)

if __name__ == "__main__":
    tok = BPETokenizer()
    fname = os.path.join(script_dir, "data", "shakespeare", "shakespeare.txt")

    with open(fname, "r") as f:
        text = f.read()
        f.close()

    # Tokenize
    tok.tokenize(text, out_vocab_size=1000)

    input_str = "O, that this too too solid flesh would melt"
    print(f"Encoding the string:\n{input_str}")
    enc_str = tok.encode(input_str)
    print(f"Encoded sequence:\n {enc_str}")
    print("\nDecoding")
    dec_str = tok.decode(enc_str)
    print(f"    {dec_str}")
