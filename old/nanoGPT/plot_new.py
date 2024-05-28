#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

"""
Plot the graphs time vs. tokens, for the generation of 2000 tokens
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

    # Create output img folder, if not present
    os.makedirs(os.path.join(script_dir, "img"), exist_ok=True)

    # Look for the files in the 'logs/tok-per-time' folder
    tok_t_folder = os.path.join(script_dir, "logs", "tok-per-time")
    assert os.path.exists(tok_t_folder), f"Error: folder not found {tok_t_folder}"

    fig = plt.figure(figsize=(6, 5))
    fnames = os.listdir(tok_t_folder)
    fnames.sort()
    for fname in fnames:
        if model_type in fname and samples_info in fname:
            # Line style
            if "mdi" in fname:
                if "2samples" in fname:
                    # label = f"MDI {model_type} 2 nodes"
                    label = "2 Nodes"
                    style = "g"
                elif "3samples" in fname:
                    # label = f"MDI {model_type} 3 nodes"
                    label = "3 Nodes"
                    style = "b"
                else:
                    label = f"MDI {model_type}"
                    style = "k"
            else:
                # label = f"Standalone {model_type}"
                label = "1 Node"
                style = "r"

            points = pd.read_csv(
                os.path.join(tok_t_folder, fname),
                sep=",",
                names=["time", "tokens"],
            )
            points_plot = points.query("tokens <= 2000")
            plt.plot(
                points_plot["tokens"],
                points_plot["time"],
                style,
                label=label,
                linewidth=2,
            )

    plt.ylabel("Time (s)")
    plt.xlabel("Number of tokens")
    plt.grid()
    plt.grid(which="minor", linestyle="dashed", linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    if "12layers" in model_type:
        title_str = "12 Layers"
    elif "9layers" in model_type:
        title_str = "9 Layers"
    elif "7layers" in model_type:
        title_str = "7 Layers"
    else:
        title_str = ""
    if title_str != "":
        plt.title(f"Standalone generation vs. MDI - {title_str}")
    else:
        plt.title(f"Standalone generation vs. MDI")
    plt.tight_layout()
    plt.savefig(
        os.path.join(script_dir, "img", f"time_vs_tokens_{model_type}.png"), dpi=500
    )
    plt.show()
