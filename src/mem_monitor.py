#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time

import GPUtil
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


def monitor_memory(process, out_file, interval=1):
    """
    Store the measured memory usage on output file.

    Args:
        process: subprocess to monitor
        out_file: pointer to an open file where to store output (readings)
        interval: time between each measurement in seconds
    """
    header = False
    gpu_list = GPUtil.getGPUs()
    while process.poll() is None:
        if (not header) and out_file != sys.stdout:
            # Write header of csv
            out_file.write("RAM_MB,")
            for i in range(len(gpu_list)):
                out_file.write(f"GPU{i}_MB,")
            out_file.write("\n")
            header = True

        try:
            # Get overall process memory usage
            process_info = psutil.Process(process.pid)
            memory_info = process_info.memory_info()
            out_file.write(f"{memory_info.rss / (1024 ** 2):.2f},")

            # Get GPU memory usage
            for gpu in gpu_list:
                out_file.write(f"{gpu.memoryUsed},")

            out_file.write("\n")

        except psutil.NoSuchProcess:
            break

        # Delay for the specified interval
        time.sleep(interval)


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
        type=int,
        default=1,
        help="time between each measurement, in seconds",
    )

    args = parser.parse_args()

    proc = start_program(args.cmd)
    monitor_memory(
        proc,
        out_file=args.output,
    )


if __name__ == "__main__":
    main()
