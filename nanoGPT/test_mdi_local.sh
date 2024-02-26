#!/bin/bash

# Test the distributed model on a single device and record times
# Perform a number of runs with the same model - the "name" is the 1st command
# line arg and the number of iterations is the 2nd one
# Example usage:
#       ./test_mdi_local.sh 7layers 100

# Clean shutdown (catch dangling processes - finisher and intermediate)
trap 'kill $(pgrep -f 'finisher.py'); kill $(pgrep -f 'intermediate.py'); kill $(pgrep -f 'starter.py')' INT

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
# Differentiate based on the number of nodes (for now: 2 or 3)
n_nodes=${3:3}
if [ "${n_nodes}" -eq 3 ];
then
    config_file="$(dirname "${0}")"/settings_distr/configuration.json
elif [ "${n_nodes}" -eq 2 ];
then
    config_file="$(dirname "${0}")"/settings_distr/configuration_2.json
fi

for i in $(seq 1 "${n_iter}");
do
    echo "Launching run number ${i}"
    python3 "$(dirname "${0}")"/finisher.py &
    python3 "$(dirname "${0}")"/intermediate.py &
    python3 "$(dirname "${0}")"/starter.py \
        --ckpt="$(dirname "${0}")"/data/shakespeare/out/ckpt_"${1}".pt \
        --time-run="$(dirname "${0}")"/logs/run_times_mdi_single_"${1}".csv \
        --nodes-config="${config_file}"
done

kill $(pgrep -f 'starter.py')
kill $(pgrep -f 'intermediate.py')
kill $(pgrep -f 'finisher.py')
