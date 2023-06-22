#!/bin/sh

#for i in 1 16 32 48 64 80 96 112 128
for i in 1
do
  echo Starting work on $i processors
  torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost --nnodes=1 --nproc_per_node=$i train.py  
done
