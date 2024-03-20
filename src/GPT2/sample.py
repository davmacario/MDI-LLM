#!/usr/bin/env python3

"""
Perform inference on a pre-trained model
"""

import cProfile
import os
import pickle
import pstats
import time
from contextlib import nullcontext
from datetime import datetime

import tiktoken
import torch

from sub import BPETokenizer, CharacterTokenizer
from sub.config import DEVICE, DTYPE, INIT_FROM, TEMPERATURE, TOP_K
from sub.model import GPT, GPTConfig
from sub.parser import parse_args
from sub.utils import count_model_layers, find_eot, plot_tokens_per_time

script_dir = os.path.dirname(__file__)

PROFILE = True


def main():
    # --------------------------------------------------------------------------
    dataset = "shakespeare"
    dataset_name = os.path.splitext(dataset)[0]
    data_dir = os.path.join(script_dir, "data", dataset_name)

    # Parse command line args
    args = parse_args(train=False)

    if args.prompt.startswith("FILE:"):
        with open(args.prompt[5:], "r") as f:
            start = f.read()
    else:
        start = args.prompt

    # Default values: 3 samples, 1000 tokens each
    num_samples = args.n_samples  # number of samples to draw
    max_new_tokens = args.n_tokens  # number of tokens generated in each sample
    seed = 1337

    using_huggingface = False

    if args.ckpt is not None:
        if args.ckpt in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}:
            print(f"Fetching model {args.ckpt} from Huggingface")
            using_huggingface = True
        else:
            assert os.path.exists(args.ckpt)
            ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(data_dir, "out", "ckpt.pt")

    VERB = args.verb
    global PROFILE
    PROFILE = args.debug

    PLOTS = args.plots

    out_stats_file = args.time_run
    if out_stats_file is not None:
        assert os.path.exists(os.path.dirname(out_stats_file))

    # --------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # for later use in torch.autocast:
    if "cuda" in DEVICE:
        device_type = "cuda"
    elif "mps" in DEVICE:
        device_type = "mps"
    else:
        device_type = "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[DTYPE]
    ctx = (  # Use autocast if on cuda or cpu (MPS not supported yet)
        nullcontext()
        if device_type == "mps"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )

    # model
    if not using_huggingface:
        # init from a model saved in a specific directory
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        gptconf = GPTConfig(**checkpoint["model_args"])

        # Note: remove first 4 chars ("ckpt"), not the underscore (there may not be)
        model_type = os.path.basename(ckpt_path).split(".")[0][4:]
        if "ctx" not in model_type:
            model_type += f"_{gptconf.block_size}ctx"

        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        n_model_layers = count_model_layers(state_dict)
        if VERB:
            print(f"Using model with {n_model_layers} layers")
        unwanted_prefix = "_orig_mod."  # NOTE: this shouldn't happen anymore
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        try:
            model.load_state_dict(state_dict)  # May need to use strict=False
        except RuntimeError:
            # Catch error thrown by 'load_state_dict'
            # This needs to be done because I removed the .attn.bias (triangular mask)
            # from the state dictionary (since it is a constant), as done by the
            # pretrained models
            # This allows me to download the models and store them locally/chunk them up
            # with reduced memory overhead (no need to create duplicate of GPT2 classes)
            missing_k, unexp_k = model.load_state_dict(state_dict, strict=False)
            # For what said above, we just allow for some keys to be unexpected (bias)
            # but if keys are missing there is a problem
            if len(missing_k) > 0:
                raise RuntimeError(f"The model is missing {len(missing_k)} keys")

    else:
        model = GPT.from_pretrained(args.ckpt, dict(dropout=0.0))
        gptconf = model.config
        model_type = args.ckpt
        n_model_layers = gptconf.n_layer
        checkpoint = None

    model.to(DEVICE)
    # if COMPILE:
    #     model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    # Look for the meta pickle in case it is available in the dataset folder
    load_char = False
    load_bpe = False
    meta_path = None
    vocab_path = None
    merges_path = None
    if checkpoint is not None:
        if (
            not using_huggingface
            and "config" in checkpoint
            and "DATASET" in checkpoint["config"]
        ):
            dataset_name = os.path.basename(
                os.path.normpath(checkpoint["config"]["DATASET"])
            )
            # Char
            meta_path = os.path.join(script_dir, "data", dataset_name, "meta.pkl")
            # BPE
            vocab_path = os.path.join(script_dir, "data", dataset_name, "encoder.json")
            merges_path = os.path.join(script_dir, "data", dataset_name, "merges.bpe")
            if os.path.exists(meta_path):
                # Use char token
                load_char = True
            elif os.path.exists(vocab_path) and os.path.exists(merges_path):
                # Use BPE token
                load_bpe = True
        elif (
            not using_huggingface
            and "config" in checkpoint
            and "DATASET_PATH" in checkpoint["config"]
        ):
            pass

    # Free up memory
    checkpoint = None

    if load_char and meta_path is not None:
        if VERB:
            print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        tok = CharacterTokenizer(meta["stoi"], meta["itos"])
        encode = tok.encode
        decode = tok.decode
    elif load_bpe and vocab_path is not None:
        if VERB:
            print(f"Loading BPE tokenizer from:\n\t{vocab_path}\n\t{merges_path}...")
        tok = BPETokenizer(vocab_path, merges_path)
        encode = tok.encode
        decode = tok.decode
    else:
        # Assume gpt-2 encodings by default FIXME
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # ---- GENERATION -------------------------------------------------------------
    # Encode the beginning of the prompt
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...]

    # Set evaluation mode
    model.eval()

    # Run generation
    tok_time_all = []
    with torch.inference_mode():
        with ctx:
            if VERB:
                print("Beginning generation")
            t_start = time.time()
            for k in range(num_samples):
                t_start_sample = time.time()
                y, tok_time = model.generate(
                    x, max_new_tokens, temperature=TEMPERATURE, top_k=TOP_K
                )
                if PLOTS:
                    tok_time_all.append(
                        [
                            (
                                x[0] + k * max_new_tokens,
                                x[1] + t_start_sample - t_start,
                            )
                            for x in tok_time
                        ]
                    )
                decoded_text = decode(y[0].tolist())
                print(decoded_text[: find_eot(decoded_text)])
                print("---------------")

    tot_gen_time = time.time() - t_start
    if VERB:
        print(f"Total generation time: {tot_gen_time} s")

    if PLOTS:
        # Store points on csv file
        points_file_path = os.path.join(
            script_dir,
            "logs",
            "tok-per-time",
            f"tokens_time_samples_standalone{model_type}_{num_samples}samples.csv",
        )
        if not os.path.exists(os.path.dirname(points_file_path)):
            os.mkdir(os.path.dirname(points_file_path))
        with open(points_file_path, "w") as f:
            for tok_t_lst in tok_time_all:
                times = [x[1] for x in tok_t_lst]
                n_samples = [x[0] for x in tok_t_lst]
                for i in range(len(times)):
                    f.write(f"{times[i]},{n_samples[i]}\n")

        # Plot tokens/time
        plot_tokens_per_time(
            tok_time_all,
            out_path=os.path.join(
                script_dir, "img", f"tokens_time_standalone{model_type}.png"
            ),
        )

    if out_stats_file is not None:
        # Output csv
        existed = True
        if not os.path.exists(out_stats_file):
            existed = False
        with open(out_stats_file, "a") as f:
            curr_ts = datetime.now()
            if not existed:
                # header
                f.write(
                    ",".join(
                        [
                            "timestamp",
                            "n_samples",
                            "n_layers",
                            "context_size",
                            "gen_time",
                        ]
                    )
                    + "\n"
                )
            f.write(
                f"{curr_ts.strftime('%Y-%m-%d %H:%M:%S')},{num_samples},{n_model_layers},{gptconf.block_size},{tot_gen_time}\n"
            )


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    if PROFILE:
        stats.print_stats()
    stats.dump_stats(os.path.join(script_dir, "logs", "sample_profile.prof"))
