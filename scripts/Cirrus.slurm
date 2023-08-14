#!/bin/bash

#SBATCH --job-name=PPO-sc2-agent
#SBATCH --time=1:00:00
#SBATCH --nodes=1

#SBATCH --account=m22oc-s2402307
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --exclusive


module load nvidia/nvhpc/22.11


cd $SLURM_SUBMIT_DIR

source sc_virt/bin/activate
export SC2PATH="${SLURM_SUBMIT_DIR}/SC2.4.9.3/StarCraftII"


# Enforce threading to 1 in case underlying libraries are threaded
export OMP_NUM_THREADS=1

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

srun python evaluate.py
#srun --ntasks=$SLURM_JOB_NUM_NODES --tasks-per-node=1 torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train.py
