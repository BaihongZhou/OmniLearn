#!/bin/bash
# First, merge the nominal samples
export PYTHONPATH=/pscratch/sd/b/baihong/python_libs/omni:$PYTHONPATH

cd syst_process
shifter python syst_merge_syst.py
wait
# After merging, produce the nominal samples
training_channel_lists=("pi_pi" "pi_rho" "rho_rho" "e_pi" "e_rho" "mu_rho" "mu_pi")
for channel in "${training_channel_lists[@]}"
do
    srun --mpi=pmi2 -n 16 -G 16 shifter --module=gpu,nccl-plugin python evaluate.py  --sample --layer_scale --local --train_channel $channel
    wait
done

# for channel in "${training_channel_lists[@]}"
# do
# 	    srun --mpi=pmi2 -n 16 -G 16 shifter --module=gpu,nccl-plugin python evaluate_1.py  --sample --layer_scale --local --train_channel $channel
# 	        wait
# 	done
# After producing the nominal samples, check the nominal samples
# cd ../valid_check
# for channel in "${training_channel_lists[@]}"
# do
#     shifter --image=vmikuni/tensorflow:ngc-23.12-tf2-v1 python valid_check.py --test_channel $channel --check_level "nominal"
#     wait
# done
