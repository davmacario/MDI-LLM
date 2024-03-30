#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Plot the memory vs. time graphs 
Example usage:
    python3 plot_mem.py gpt2-medium 3
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
parser.add_argument(
    "N_NODES",
    type=int,
    help="""Number of nodes for the plot, i.e., how many nodes were used in the run and
    will be present in the graph.""",
)

if __name__ == "__main__":
    args = parser.parse_args()
    model_type = args.MODEL

    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

    nodes_configuration = {1: "single", 2: "2nodes", 3: "3nodes"}[args.N_NODES]

    # Create output img folder, if not present
    os.makedirs(os.path.join(script_dir, "img"), exist_ok=True)

    # Look for the files in the 'logs/tok-per-time' folder
    mem_t_folder = os.path.join(script_dir, "logs", "mem-usage", model_type)
    assert os.path.exists(mem_t_folder), f"Error: folder not found {mem_t_folder}"

    fig = plt.figure(figsize=(6, 5))
    fnames = os.listdir(mem_t_folder)
    fnames.sort()
    for fname in fnames:
        if nodes_configuration in fname:
            # Line style
            if "starter" in fname:
                label = "First node"
                style = "r"
            elif "secondary0" in fname:
                # label = f"MDI {model_type} 2 nodes"
                label = "Second node"
                style = "b"
            elif "secondary1" in fname:
                # label = f"MDI {model_type} 3 nodes"
                label = "Third node"
                style = "g"
            else:
                print("Warning: shouldn't be here!")
                exit(0)

            points = pd.read_csv(
                os.path.join(mem_t_folder, fname),
                sep=",",
            )
            time_res = 0.5
            time = np.arange(0, time_res * len(points), time_res)
            plt.plot(
                time,
                points["RAM_MB"],
                style,
                label=label,
                linewidth=2,
            )

    plt.ylabel("Memory usage (MB)")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.grid(which="minor", linestyle="dashed", linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.title(f"Memory usage, {args.N_NODES} node(s), {model_type}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            script_dir, "img", f"mem_in_time_{model_type}_{args.N_NODES}nodes.png"
        ),
        dpi=500,
    )
    plt.show()
