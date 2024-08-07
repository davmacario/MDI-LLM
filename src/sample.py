#!/usr/bin/env python3

"""
Perform inference on a pre-trained model - TinyLlama & Llama
"""

import cProfile
import gc
import os
import pstats
import time
import warnings
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path

import torch

from sub import GPT, PromptStyle, Tokenizer
from sub.config import DTYPE_TORCH_MAPPING, TEMPERATURE, TOP_K
from sub.prompts import get_user_prompt, has_prompt_style, load_prompt_style
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

    batch_size = args.n_samples  # number of samples to draw
    using_huggingface = False

    dtype = (
        (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        if args.dtype is None
        else args.dtype
    )
    if dtype not in DTYPE_TORCH_MAPPING.keys():
        raise ValueError(
            f"Unknown dtype {dtype}, supported ones are: {DTYPE_TORCH_MAPPING.keys()}"
        )
    if dtype == "bfloat16" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError(
            "Specified bfloat16, but the host does not support this format"
        )
    ptdtype = DTYPE_TORCH_MAPPING[dtype]

    checkpoint_dir = Path(args.ckpt)
    model_type = checkpoint_dir.name
    checkpoint_path = args.ckpt / "lit_model.pth"  # Requires ckpt to be converted
    if not checkpoint_path.is_file() and (
        (checkpoint_dir / "model.bin").is_file()
        or (checkpoint_dir / "pytorch_model.bin.index.json").is_file()
        or (checkpoint_dir / "model.safetensors.index.json").is_file()
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
    if "cuda" in args.device:
        device_type = "cuda"
    elif "mps" in args.device:
        device_type = "mps"
    else:
        device_type = "cpu"
    if args.verb:
        print(f"Using {args.device}")
        print(f"Device type: {device_type}")
        print(f"Dtype: {dtype}")
    ctx = (  # Use autocast if on cuda or cpu (MPS not supported yet)
        nullcontext()
        if device_type == "mps" or not dtype == "bfloat16"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )
    torch_device = torch.device(args.device)

    # Model setup
    config, wt = load_from_pt(checkpoint_dir)
    assert wt is not None
    model = GPT(config)

    if args.sequence_length:
        print(f"Truncating model context size to {args.sequence_length}")
        model.max_seq_length = args.sequence_length

    if args.dtype:
        model_dtype = DTYPE_TORCH_MAPPING[args.dtype]
    else:
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

    # model.set_kv_cache(batch_size=batch_size)  # process samples together
    model.set_kv_cache(
        batch_size=1, device=torch_device
    )  # Re-set cache for every sample

    # Compile model + catch exception if unsupported (Python 3.12 currently)
    if args.compile and hasattr(torch, "compile"):
        if args.verb:
            print("Compiling model - this may take a while", end="\r")
        try:
            model = torch.compile(model)
            if args.verb:
                print("Model compiled!")
        except RuntimeError as e:
            warnings.warn(f"Unable to compile model! {e}")
    elif args.compile and not hasattr(torch, "compile"):
        from importlib.metadata import version

        warnings.warn(
            f"Installed torch version ({version('torch')}) does not support compiling models"
        )

    model.eval()

    # Tokenizer
    try:
        tokenizer = Tokenizer(checkpoint_dir, force_backend="huggingface")
    except:
        tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir)
        if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )
    stop_tokens = prompt_style.stop_tokens(tokenizer)
    start = get_user_prompt(args.prompt, batch_size, prompt_style)

    # ---- GENERATION -------------------------------------------------------------
    # Encode the prompt
    # Run generation
    tok_time_all = []
    with ctx, torch.inference_mode():
        if args.verb:
            print("Beginning generation")
        t_start = time.time()
        for k in range(batch_size):
            curr_tok_time = []
            t_start_sample = time.time()
            prompt = start[k]
            if args.verb:
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
                    (x[0] + k * (args.n_tokens), x[1] + t_start_sample - t_start)
                    for x in curr_tok_time
                ]
            )
            truncated = find_eot(y, stop_tokens, len(start_ids))
            if args.verb:
                print(
                    f"Output was truncated to {len(truncated.squeeze())}/{len(y.squeeze())} tokens"
                )
            decoded_text = tokenizer.decode(truncated)
            print(decoded_text)
            print("---------------")

            for block in model.transformer.h:
                block.attn.kv_cache.reset_parameters()

    tot_gen_time = time.time() - t_start
    if args.verb:
        print(f"Total generation time: {tot_gen_time} s")

    if args.plots:
        # Store points on csv file
        os.makedirs(os.path.join(script_dir, "logs"), exist_ok=True)
        points_file_path = os.path.join(
            script_dir,
            "logs",
            f"tokens_time_samples_1nodes_{model_type}_{batch_size}samples.csv",
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
                f"tokens_time_1nodes_{model_type}_{batch_size}samples.png",
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
    #             f"{curr_ts.strftime('%Y-%m-%d %H:%M:%S')},{batch_size},{n_model_layers},{gptconf.block_size},{tot_gen_time}\n"
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
        "-c",
        "--compile",
        action="store_true",
        help="if set, compile the model (Torch >= 2.0.0 required)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=script_dir / "checkpoints",
        help=f"folder containing model files (default={script_dir / 'checkpoints'})",
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
        "--sequence-length",
        "--context-length",
        "--block-size",
        type=int,
        default=None,
        help="""
        sequence length of the model, i.e., maximum span of the attention window;
        if not specified, it will use the default model sequence length;
        allows to reduce RAM usage, as with a shorter context less cache is created.
        """,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="""the model dtype (among float32, float16 and bfloat16 - if supported)""",
    )
    parser.add_argument("--seed", type=int, default=10137, help="set random seed")

    args = parser.parse_args()

    main(args)
