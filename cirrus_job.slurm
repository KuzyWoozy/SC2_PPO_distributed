#!/bin/bash

#SBATCH --job-name=rl-agent
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=m22oc-s2402307
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source torch-env/bin/activate

srun python train.py
