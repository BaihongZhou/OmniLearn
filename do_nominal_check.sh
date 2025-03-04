#!/bin/bash
# First, merge the nominal samples
export PYTHONPATH=/pscratch/sd/b/baihong/python_libs/omni:$PYTHONPATH

cd syst_process
shifter python syst_merge.py
wait
# After merging, produce the nominal samples
training_channel_lists=("pi_pi" "e_pi" "mu_pi")
for channel in "${training_channel_lists[@]}"
do
    srun --mpi=pmi2 -n 16 -G 16 shifter --module=gpu,nccl-plugin python evaluate_nominal.py  --sample --layer_scale --local --train_channel $channel
    wait
done