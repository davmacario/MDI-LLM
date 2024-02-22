#!/bin/bash

# Test model on a single device and record times
# Perform a number of runs with the same model - the "name" is the 1st command
# line arg and the number of iterations is the 2nd one
# Example usage:
#       ./test_local_gen.sh 7layers 100

# Clean shutdown (catch dangling processes - finisher and intermediate)
trap 'kill $(pgrep -f 'sample.py')' INT

if [ "$#" -eq 0 ];
then
    echo "ERROR: missing model type/name in args"
    exit 1
fi
if [ ! -f "$(dirname "${0}")/data/shakespeare/out/ckpt_${1}.pt" ];
then
    echo "ERROR: $(dirname "${0}")/data/shakespeare/out/ckpt_${1}.pt does not exist"
    exit 2
fi

n_iter=${2:-10}
for i in $(seq 1 "${n_iter}");
do
    echo "Launching run number ${i}"
    python3 "$(dirname "${0}")"/sample.py \
        --ckpt="$(dirname "${0}")"/data/shakespeare/out/ckpt_"${1}".pt \
        --time-run="$(dirname "${0}")"/logs/run_times_single_"${1}".csv \
        --verb
done
