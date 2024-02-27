#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

script_dir = os.path.dirname(__file__)

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
    for fname in os.listdir(tok_t_folder):
        # Ugly :/
        if model_type in fname and samples_info in fname:
            label = "MDI" if "mdi" in fname else "Standalone"
            style = "b" if "mdi" in fname else "r"
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
