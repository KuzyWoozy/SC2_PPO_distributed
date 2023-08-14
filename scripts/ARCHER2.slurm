#!/bin/bash

#SBATCH --job-name=PPO-sc2-agent
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --account=m22oc-s2402307
#SBATCH --partition=standard
#SBATCH --qos=standard

source ../sc_virt/bin/activate

export SC2PATH="/work/m22oc/m22oc/s2402307/StarCraftII"

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Enforce threading to 1 in case underlying libraries are threaded
export OMP_NUM_THREADS=1


export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

srun torchrun --nnodes=1 --nproc_per_node=$SLURM_CPUS_PER_TASK --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train.py
