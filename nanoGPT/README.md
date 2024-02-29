# Model-Distributed Inference – NanoGPT

This folder contains the implementation of MDI for A. Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

The main blocks have been re-implemented from scratch, starting from his YouTube [video](https://youtu.be/kCc8FmEb1nY?si=WuCzAHZK54VdoSLT).
The rationale was to keep the blocks as modular as possible, allowing to split them across multiple devices (Nvidia Jetson TX2).

Additional features have been added to make debugging easier (command line argument parsing, logger).

Training and inference script have been adapted from the original implementation

## Folder structure

- Training script: [`./train.py`](./train.py)
- Inference script: [`./sample.py`](./sample.py)
- Set-up script: [`./prepare_data.py`](./prepare_data.py) (encode text file and split in train & test set)
- Model library: [`sub` folder](./sub)
  - Original (rewritten) nanoGPT implementation: [`./sub/model.py`](./sub/model.py)
  - <mark>MDI implementation:</mark> [`./sub/model_dist.py`](./sub/model_dist.py)
  - Global configuration file [`./sub/config.py`](./sub/config.py)
  - Parser (one size fits all): [`./sub/parser.py`](./sub/parser.py)
  - Character-based tokenizer: [`./sub/char_tokenizer.py`](./sub/char_tokenizer.py)
  - BPE tokenizer (custom implementation to play with vocabulary size): [`./sub/bpe_tokenizer.py`](./sub/bpe_tokenizer.py)
  - Data-loading utilities: [`./sub/data_loader.py`](./sub/data_loader.py)
  - Utility functions [`./sub/utils.py`](./sub/utils.py) (e.g., splitting pretrained model, loading bar)
- Entrypoints for MDI execution:
  - Starter node: [`./starter.py`](./starter.py)
  - Intermediate node: [`./intermediate.py`](./intermediate.py)
  - Finisher node: [`./finisher.py`](./finisher.py)
- Testing programs:
  - Testing checkpoints (pretrained models) structure: [`./test_checkpoint.py`](./test_checkpoint.py)
  - Testing tokenizer: [`./test_tokenizer.py`](./test_tokenizer.py)
  - Test local generation (multi-run): [`./test_local_gen.sh`](./test_local_gen.sh)
  - Test MDI (multi-run): [`./test_mdi_local.sh`](./test_mdi_local.sh)
