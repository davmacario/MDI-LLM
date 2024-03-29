#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

"""
Plot the graphs time vs. tokens
Example usage:
    python3 plot_tok_time.py gpt2-medium
"""

script_dir = os.path.dirname(__file__)
line_styles = ["-", "--", "-.", ":"]

# Need to check that "_'key'_" is in the title
map_model_name = {
    "gpt2": "12layers",
    "gpt2-medium": "24layers",
    "gpt2-large": "36layers",
    "gpt2-xl": "48layers",
}

parser = argparse.ArgumentParser(
    description="""Display number of generated tokens vs. time for the specified model"""
)
parser.add_argument(
    "MODEL",
    type=str,
    help="The model whose results should be plotted, e.g., 'gpt2', 'gpt2-medium'",
)

if __name__ == "__main__":
    args = parser.parse_args()
    model_type = args.MODEL

    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

    # Create output img folder, if not present
    os.makedirs(os.path.join(script_dir, "img"), exist_ok=True)

    # Look for the files in the 'logs/tok-per-time' folder
    tok_t_folder = os.path.join(script_dir, "logs", "tok-per-time", model_type)
    assert os.path.exists(tok_t_folder), f"Error: folder not found {tok_t_folder}"

    fig = plt.figure(figsize=(6, 5))
    fnames = os.listdir(tok_t_folder)
    fnames.sort()
    for fname in fnames:
        if model_type in fname:
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
                    print("Warning: shouldn't be here!")
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
            points_plot = points.query("tokens <= 800")
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
    plt.title(f"Standalone generation vs. MDI - {model_type}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(script_dir, "img", f"time_vs_tokens_{model_type}.png"), dpi=500
    )
    plt.show()
