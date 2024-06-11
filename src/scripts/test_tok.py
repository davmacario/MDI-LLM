from argparse import ArgumentParser
from pathlib import Path
from tokenizers import Tokenizer

def main(args):
    tok = Tokenizer.from_file(str(args.PATH / "tokenizer_json"))

    bos_id = tok.bos_id()
    eos_id = tok.eos_id()

    print(f"Beginning of sentence: {bos_id} -> {repr(tok.decode([bos_id]))}")
    print(f"End of sentence: {eos_id} -> {repr(tok.decode([eos_id]))}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Inspect a sentencepiece tokenizer")
    parser.add_argument("PATH", type=Path, help="folder containing tokenizer files")
    args = parser.parse_args()
    main(args)
