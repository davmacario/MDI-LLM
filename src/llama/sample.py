#!/usr/bin/env python3

"""
Perform inference on a pre-trained model - TinyLlama & Llama
"""

import cProfile
import os
import pickle
import pstats
import time
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tiktoken
import torch
from sub import GPT, Config, PromptStyle, Tokenizer
from sub.config import DTYPE, TEMPERATURE, TOP_K  # TODO: change dtype def
from sub.prompts import has_prompt_style, load_prompt_style
from sub.utils import count_model_layers, find_eot, plot_tokens_per_time

script_dir = Path(os.path.dirname(__file__))


def main(args):
    # Set up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn

    if args.debug:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.prompt.startswith("FILE:"):
        with open(args.prompt[5:], "r") as f:
            start = f.read()
    else:
        start = args.prompt

    BATCH_SIZE = args.n_samples  # number of samples to draw

    using_huggingface = False

    VERB = args.verb
    PLOTS = args.plots
    DEVICE = args.device

    checkpoint_dir = args.ckpt
    checkpoint_path = args.ckpt / "lit_model.pth"  # Requires ckpt to be converted
    if not checkpoint_path.is_file() and (
        (checkpoint_dir / "model.bin").is_file()
        or (checkpoint_dir / "pytorch_model.bin.index.json").is_file()
    ):
        # Weights are there but in wrong format
        from sub.convert_hf_checkpoint import convert_hf_checkpoint

        convert_hf_checkpoint(checkpoint_dir=checkpoint_dir, dtype=DTYPE)

    assert checkpoint_path.is_file(), "Something went wrong in weight conversion"

    # out_stats_file = args.time_run
    # if out_stats_file is not None:
    #     assert os.path.exists(os.path.dirname(out_stats_file))

    # --------------------------------------------------------------------------
    # For later use in torch.autocast:
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

    # Model setup
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    model = GPT(config)

    # NOTE: by increasing the batch size, the model can generate more samples together
    # but this would not be fair compared to MDI, as we could raise the batch size
    # there as well; instead, we generate individual samples multiple times

    # model.set_kv_cache(batch_size=BATCH_SIZE)  # process samples together
    model.set_kv_cache(batch_size=1)  # Re-set cache for every sample

    model.eval()

    # Unsupported
    # if compile:
    #     [...]

    # Tokenizer
    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir)
        if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )
    stop_tokens = prompt_style.stop_tokens(tokenizer)



    # ---- GENERATION -------------------------------------------------------------
    # Encode the prompt
    # Run generation
    tok_time_all = []
    with torch.inference_mode():
        with ctx:
            if VERB:
                print("Beginning generation")

            t_start = time.time()
            for k in range(BATCH_SIZE):
                # TODO: fix support for one prompt per sample
                prompt = prompt_style.apply(start)
                start_ids = tokenizer.encode(prompt, device=DEVICE)
                # Ensure the desired amount of new tokens is generated
                max_new_tokens = start_ids.size(0) + args.n_tokens

                x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...]

                t_start_sample = time.time()
                y = model.generate(
                    x, max_new_tokens, temperature=TEMPERATURE, top_k=TOP_K
                )
                decoded_text = tokenizer.decode(y[0].tolist())
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
            f"tokens_time_samples_standalone{model_type}_{BATCH_SIZE}samples.csv",
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

    # if out_stats_file is not None:
    #     # Output csv
    #     existed = True
    #     if not os.path.exists(out_stats_file):
    #         existed = False
    #     with open(out_stats_file, "a") as f:
    #         curr_ts = datetime.now()
    #         if not existed:
    #             # header
    #             f.write(
    #                 ",".join(
    #                     [
    #                         "timestamp",
    #                         "n_samples",
    #                         "n_layers",
    #                         "context_size",
    #                         "gen_time",
    #                     ]
    #                 )
    #                 + "\n"
    #             )
    #         f.write(
    #             f"{curr_ts.strftime('%Y-%m-%d %H:%M:%S')},{BATCH_SIZE},{n_model_layers},{gptconf.block_size},{tot_gen_time}\n"
    #         )

    if profiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats()
        stats.dump_stats(os.path.join(script_dir, "logs", "sample_profile.prof"))


if __name__ == "__main__":
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    parser = ArgumentParser(description="""LLM inference""")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable debug mode (profiler)"
    )
    parser.add_argument("-v", "--verb", action="store_true", help="enable verbose mode")
    parser.add_argument("-p", "--plots", action="store_true", help="enable plots")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=script_dir / "checkpoint",
        help="folder containing the model files",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="batch size, i.e., n. of generated samples, i.e., produced pieces of text",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=800,
        help="number of generated tokens per sample, excluding the prompt",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=default_device,
        help="torch device where to load model and tensors",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="\n",
        help="""specify a prompt for the language model;
        if starting with 'FILE:', the prompt will be extracted for a file.
        If the string is the prompt itself, it will be used for all generated samples,
        while if a file is specified, each paragraph (separated by blank line) will be 
        used for a different sample, with extra prompts being discarded if --n-samples
        is lower than the number of paragraphs.
        """,
    )
    parser.add_argument("--seed", type=int, default=10137, help="set random seed")

    args = parser.parse_args()

    main(args)
