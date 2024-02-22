#!/bin/bash

# Test model on a single device and record times
# Perform 10 runs with the same model - the "name" is the 1st command line arg
# Example usage:
#       ./test_local_gen.sh 7layers

if [ "$#" -eq 0 ];
then
    echo "ERROR: missing model type/name in args"
    exit 1
fi

if [ ! -f "$(dirname $0)"/data/shakespeare/out/ckpt_"$1".pt ];
then
    echo "ERROR: $(dirname ${0})/data/shakespeare/out/ckpt_${1}.pt} does not exist"
    exit 2
fi

for i in $(seq 1 10);
do
    echo "Launching run number ${i}"
    python3 "$(dirname ${0})"/finisher.py &
    python3 "$(dirname ${0})"/intermediate.py &
    python3 "$(dirname ${0})"/starter.py \
        --ckpt="$(dirname ${0})"/data/shakespeare/out/ckpt_"${1}".pt \
        --time-run="$(dirname ${0})"/logs/run_times_mdi_single_"${1}".csv
done
