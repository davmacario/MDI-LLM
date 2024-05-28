#!/usr/bin/env python3

"""
Perform inference on a pre-trained model - TinyLlama & Llama
"""

import cProfile
import os
import pstats
import time
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
import gc

import torch
from sub.config import TEMPERATURE, TOP_K
from sub.prompts import get_user_prompt, has_prompt_style, load_prompt_style

from sub import GPT, PromptStyle, Tokenizer
from sub.utils import find_eot, load_from_pt, plot_tokens_per_time

script_dir = Path(os.path.dirname(__file__))


def main(args):
    # Set up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn

    profiler = None
    if args.debug:
        profiler = cProfile.Profile()
        profiler.enable()

    BATCH_SIZE = args.n_samples  # number of samples to draw

    using_huggingface = False

    VERB = args.verb
    PLOTS = args.plots
    DEVICE = args.device

    checkpoint_dir = Path(args.ckpt)
    model_type = checkpoint_dir.name
    checkpoint_path = args.ckpt / "lit_model.pth"  # Requires ckpt to be converted
    if not checkpoint_path.is_file() and (
        (checkpoint_dir / "model.bin").is_file()
        or (checkpoint_dir / "pytorch_model.bin.index.json").is_file()
    ):
        # Weights are there but in wrong format
        from sub.utils.convert_hf_checkpoint import convert_hf_checkpoint

        convert_hf_checkpoint(checkpoint_dir=checkpoint_dir, dtype=dtype)

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
    if VERB:
        print(f"Using {DEVICE}")
        print(f"Device type: {device_type}")
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (  # Use autocast if on cuda or cpu (MPS not supported yet)
        nullcontext()
        if device_type == "mps"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )
    torch_device = torch.device(DEVICE)

    # Model setup
    config, wt = load_from_pt(checkpoint_dir)
    assert wt is not None
    model = GPT(config)

    model_dtype = torch.float32
    if all([v.dtype == torch.float16 for v in wt.values()]):
        model_dtype = torch.float16
    elif all([v.dtype == torch.bfloat16 for v in wt.values()]):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16

    if model_dtype in {torch.float16, torch.bfloat16}:
        model = model.to(model_dtype)
    model.load_state_dict(wt)
    del wt
    gc.collect()
    model.to(torch_device)

    n_params = sum(p.numel() for p in model.parameters())
    if args.verb:
        print(f"Number of model parameters: {n_params:,}")

    # NOTE: by increasing the batch size, the model can generate more samples together
    # but this would not be fair compared to MDI, as we could raise the batch size
    # there as well; instead, we generate individual samples multiple times

    # model.set_kv_cache(batch_size=BATCH_SIZE)  # process samples together
    model.set_kv_cache(
        batch_size=1, device=torch_device
    )  # Re-set cache for every sample

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
    start = get_user_prompt(args.prompt, BATCH_SIZE, prompt_style)

    # ---- GENERATION -------------------------------------------------------------
    # Encode the prompt
    # Run generation
    tok_time_all = []
    with ctx:
        if VERB:
            print("Beginning generation")
        t_start = time.time()
        for k in range(BATCH_SIZE):
            curr_tok_time = []
            t_start_sample = time.time()
            prompt = start[k]
            if VERB:
                print(prompt)
            start_ids = tokenizer.encode(prompt, device=torch_device)
            # Ensure the desired amount of new tokens is generated
            max_new_tokens = start_ids.size(0) + args.n_tokens

            y = model.generate(
                start_ids,
                max_new_tokens,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                tok_time=curr_tok_time,
            )
            tok_time_all.append(
                [
                    (x[0] + k * max_new_tokens, x[1] + t_start_sample - t_start)
                    for x in curr_tok_time
                ]
            )
            decoded_text = tokenizer.decode(find_eot(y, stop_tokens))
            print(decoded_text)
            print("---------------")

    tot_gen_time = time.time() - t_start
    if VERB:
        print(f"Total generation time: {tot_gen_time} s")

    if PLOTS:
        # Store points on csv file
        os.makedirs(os.path.join(script_dir, "logs"), exist_ok=True)
        points_file_path = os.path.join(
            script_dir,
            "logs",
            f"tokens_time_samples_1nodes_{model_type}_{BATCH_SIZE}samples.csv",
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
        os.makedirs(os.path.join(script_dir, "img"), exist_ok=True)
        plot_tokens_per_time(
            tok_time_all,
            out_path=os.path.join(
                script_dir,
                "img",
                f"tokens_time_1nodes_{model_type}_{BATCH_SIZE}samples.png",
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
        default=script_dir / "checkpoints",
        help=f"folder containing model files (default={script_dir / 'checkpoints'})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="""batch size, i.e., n. of generated samples, i.e., produced pieces of 
        text (default=1)""",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=800,
        help="number of generated tokens per sample, excluding prompt (default=800)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help=f"torch device where to load model and tensors (default={default_device}",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Who are you?",
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
