#!/bin/bash

# Perform 10 runs with the same model - the "name" is the 1st command line arg
# Example usage:
#       ./test_local_gen.sh 7layers
for i in $(seq 1 10);
do
    python3 "$(dirname $0)"/sample.py --ckpt="$(dirname $0)"/data/shakespeare/out/ckpt_"$1".pt --time-run="$(dirname $0)"/logs/run_times_single_"$1".csv --verb
done
