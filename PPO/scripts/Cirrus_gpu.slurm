#!/bin/bash

#SBATCH --job-name=PPO-sc2-agent
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10

#SBATCH --account=m22oc-s2402307
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=short


module load nvidia/nvhpc/22.11

source ./sc_virt/bin/activate

# Change to the submission directory
cd $SLURM_SUBMIT_DIR
export SC2PATH="${SLURM_SUBMIT_DIR}/SC2.4.9.3/StarCraftII"

# Enforce threading to 1 in case underlying libraries are threaded
export OMP_NUM_THREADS=1

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

srun torchrun --nnodes=1 --nproc_per_node=$SLURM_CPUS_PER_TASK --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train.py
