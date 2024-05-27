#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

"""
Plot the graphs time vs. tokens
Example usage:
    python3 plot_tok_time.py gpt2-medium
"""


def main(args):
    script_dir = Path(os.path.dirname(__file__))
    img_dir = script_dir / "img"
    logs_folder = script_dir / "logs"
    if not logs_folder.exists():
        raise FileNotFoundError("Unable to locate 'logs' directory containing the data")
    model_name = args.MODEL_DIR.name

    os.makedirs(img_dir, exist_ok=True)

    fig = plt.figure(figsize=(6, 5))
    for log_file in sorted(logs_folder.iterdir()):
        fname = log_file.name
        if model_name in fname:
            if "1nodes" in fname:
                label = "1 Node"
                style = "r"
            elif "2nodes" in fname:
                label = "2 Nodes"
                style = "g"
            elif "3nodes" in fname:
                label = "3 Nodes"
                style = "b"
            else:
                # Maybe add default style
                raise FileNotFoundError

            points = pd.read_csv(log_file, sep=",", names=["time", "tokens"])
            # Use the following to crop to fixed n. of tokens - shouldn't be needed
            # points = points.query()
            plt.plot(points["tokens"], points["time"], style, label=label, linewidth=2)

    plt.ylabel("Time (s)")
    plt.xlabel("Number of tokens")
    plt.grid()
    plt.grid(which="minor", linestyle="dashed", linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    if not args.no_title:
        plt.title(f"Standalone generation vs. MDI - {model_name}")
    plt.tight_layout()
    plt.savefig(
        img_dir / f"time_vs_tokens_{model_name}.png", dpi=500
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Produce plots displaying the number of generated tokens vs. the
        time"""
    )
    parser.add_argument(
        "MODEL_DIR",
        type=Path,
        help="the directory of the model whose results will be plotted",
    )
    parser.add_argument(
        "-nt",
        "--no-title",
        action="store_true",
        help="if set, don't print the figure title"
    )
    args = parser.parse_args()
    main(args)
