#!/bin/bash

# Get this script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="${DIR}:$PYTHONPATH"


# Train
srun --mpi=pmi2 -n 16 -G 16 shifter --module=gpu,nccl-plugin python ${DIR}/scripts/evaluate.py  --config

# Evaluate
srun --mpi=pmi2 -n 16 -G 16 shifter --module=gpu,nccl-plugin python ${DIR}/scripts/train.py  --config
