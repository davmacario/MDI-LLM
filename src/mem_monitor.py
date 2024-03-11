#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil


def start_program(cmd: str):
    """
    Launch program you want to monitor the memory usage of.

    Args:
        cmd: string containing the command to be launched
    """
    cmd_as_list = cmd.split()
    process = subprocess.Popen(cmd_as_list)
    return process


def monitor_memory(process, out_file, interval=1, img_path=None):
    """
    Store the measured memory usage on output file.

    Args:
        process: subprocess to monitor
        out_file: pointer to an open file where to store output (readings)
        interval: time between each measurement in seconds
        img_path (optional): path of the output image plotting memory usage in time
    """
    header = False
    gpu_list = GPUtil.getGPUs()
    while process.poll() is None:
        if (not header) and out_file != sys.stdout:
            # Write header of csv
            out_file.write("RAM_MB")
            for i in range(len(gpu_list)):
                out_file.write(f",GPU{i}_MB")
            out_file.write("\n")
            header = True

        try:
            # Get overall process memory usage
            process_info = psutil.Process(process.pid)
            memory_info = process_info.memory_info()
            out_file.write(f"{memory_info.rss / (1024 ** 2):.2f}")

            # Get GPU memory usage
            gpu_list = GPUtil.getGPUs()
            for gpu in gpu_list:
                out_file.write(f",{gpu.memoryUsed}")

            out_file.write("\n")

        except psutil.NoSuchProcess:
            break

        # Delay for the specified interval
        time.sleep(interval)

    out_file.close()

    # Plot - if specified
    if img_path is not None and out_file != sys.stdout:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        # Read csv:
        with open(out_file.name) as f:
            df_values = pd.read_csv(f)
        rows, cols = df_values.shape
        t_axis = np.arange(0, rows * interval, interval)

        plt.figure(figsize=(12, 8))
        plt.plot(t_axis, df_values["RAM_MB"], label="RAM")
        for i in range(len(gpu_list)):
            plt.plot(t_axis, df_values[f"GPU{i}_MB"], label=f"VRAM GPU {i}")
        plt.grid()
        plt.title("Memory usage")
        plt.xlabel("Time (s)")
        plt.ylabel("Usage (MB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path)
        plt.show()


def main():
    parser = argparse.ArgumentParser(prog="Memory Monitor")
    parser.add_argument(
        "cmd",
        metavar="CMD",
        type=str,
        help="command to be executed (whose memory usage will be monitored) - note: use double quotes",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s 0.1",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="output file where to store the memory usage logs",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1,
        help="time between each measurement, in seconds",
    )
    parser.add_argument(
        "--img",
        type=str,
        help="output image name - if not specified, no image will be produced",
    )

    args = parser.parse_args()

    proc = start_program(args.cmd)
    monitor_memory(
        proc, out_file=args.output, interval=args.interval, img_path=args.img
    )


if __name__ == "__main__":
    main()
