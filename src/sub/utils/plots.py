import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt

from sub.typing import FileType

file_dir = os.path.dirname(__file__)

PlotPoints = Tuple[int, float]

def plot_tokens_per_time(
    tok_time: List[Union[PlotPoints, List[PlotPoints]]],
    out_path: FileType = os.path.join(file_dir, "..", "img", "tokens_time.png"),
    disp: bool = True,
):
    """
    Plot a graph representing the number of generated tokens in time.

    Args:
        tok_time: list of couples, where the 1st element is the number of
            samples and the 2nd element is the time at which it was generated.
            It can also be a list of list of couples (multiple samples); in this
            case, the plot will distinguish between the different samples
        out_path: path of the produced output image
        disp: if true, the image will also be displayed at runtime
    """
    assert len(tok_time) >= 1
    # Create missing dirs:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    if isinstance(tok_time[0], Tuple):
        time = [x[1] for x in tok_time]
        n_samples = [x[0] for x in tok_time]
        plt.plot(time, n_samples)
        plt.title("Number of generated samples vs. time - MDI")
    elif isinstance(tok_time[0], List):
        for i, sublist in enumerate(tok_time):
            time = [x[1] for x in sublist]
            n_samples = [x[0] for x in sublist]
            plt.plot(time, n_samples, label=f"Sample {i + 1}")
            plt.legend()
        plt.title("Number of generated samples vs. time - standalone")
    plt.xlabel("Time (s)")
    plt.ylabel("N. samples")
    plt.grid()
    plt.tight_layout()
    fig.savefig(out_path)
    if disp:
        plt.show()
