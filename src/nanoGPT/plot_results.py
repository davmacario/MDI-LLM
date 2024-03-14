#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

"""
Example usage:
    python3 plot_results.py 12layers -n 3samples
"""

script_dir = os.path.dirname(__file__)
line_styles = ["-", "--", "-.", ":"]

parser = argparse.ArgumentParser(
    description="""Display number of generated tokens vs. 
                time, comparing different settings, i.e., number of
                nodes, for the same model used"""
)
parser.add_argument(
    "MODEL",
    type=str,
    help="The model whose results should be plotted, e.g., '9layers', '12layers_128ctx'",
)
parser.add_argument(
    "-n",
    "--n-samples",
    type=int,
    default=None,
    help="If specified, only display results that generate this number of samples",
)

if __name__ == "__main__":
    args = parser.parse_args()
    model_type = args.MODEL

    # Second optional arg: specify samples
    if args.n_samples is not None:
        samples_info = f"{args.n_samples}samples"
    else:
        samples_info = ""

    # Look for the files in the 'logs/tok-per-time' folder
    tok_t_folder = os.path.join(script_dir, "logs", "tok-per-time")
    assert os.path.exists(tok_t_folder), f"Error: folder not found {tok_t_folder}"

    fig = plt.figure(figsize=(12, 6))
    for fname in os.listdir(tok_t_folder):
        if model_type in fname and samples_info in fname:
            # Line style
            if "2samples" in fname:
                line_style = line_styles[1]
                n_samples = "2 samples"
            elif "3samples" in fname:
                line_style = line_styles[0]
                n_samples = "3 samples"
            else:
                line_style = line_styles[2]
                n_samples = ""

            # Line color
            if "mdi" in fname:
                label = f"MDI {model_type} {n_samples}"
                style = "b" + line_style
            else:
                label = f"Standalone {model_type} {n_samples}"
                style = "r" + line_style

            points = pd.read_csv(
                os.path.join(tok_t_folder, fname),
                sep=",",
                names=["time", "tokens"],
            )
            plt.plot(points["time"], points["tokens"], style, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("N. tokens")
    plt.grid()
    plt.legend()
    plt.title("Comparison - standalone generation vs. MDI")
    plt.tight_layout()
    plt.savefig(
        os.path.join(script_dir, "img", f"tokens_time_comparison_{model_type}.png")
    )
    plt.show()
