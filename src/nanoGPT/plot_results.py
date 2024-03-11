#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sub.utils import remove_prefix

script_dir = os.path.dirname(__file__)
line_styles = ["-", "--", "-.", ":"]

"""
Example usage:
    python3 plot_results.py 12layers 3samples
"""

if __name__ == "__main__":
    # Parse arg: model type
    assert len(sys.argv) > 1, "Missing argument: model type"
    model_type = str(sys.argv[1])

    # Second optional arg: specify samples
    try:
        samples_info = str(sys.argv[2])
    except:
        samples_info = ""

    # Look for the files in the 'logs/tok-per-time' folder
    tok_t_folder = os.path.join(script_dir, "logs", "tok-per-time")
    assert os.path.exists(
        tok_t_folder
    ), f"Error: folder not found {tok_t_folder}"

    fig = plt.figure(figsize=(12, 6))
    ind_style_mdi = 0
    ind_style_standalone = 0
    for fname in os.listdir(tok_t_folder):
        # Ugly :/
        if model_type in fname and samples_info in fname:
            if "mdi" in fname:
                full_model_spec = remove_prefix(
                    os.path.splitext(fname)[0], "tokens_time_samples_mdi_"
                )
                label = f"MDI {full_model_spec}"
                style = "b" + line_styles[ind_style_mdi]
                ind_style_mdi = (ind_style_mdi + 1) % len(line_styles)
            else:
                full_model_spec = remove_prefix(
                    os.path.splitext(fname)[0],
                    "tokens_time_samples_standalone_",
                )
                label = f"Standalone {full_model_spec}"
                style = "r" + line_styles[ind_style_standalone]
                ind_style_standalone = (ind_style_standalone + 1) % len(
                    line_styles
                )

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
        os.path.join(
            script_dir, "img", f"tokens_time_comparison_{model_type}.png"
        )
    )
    plt.show()
