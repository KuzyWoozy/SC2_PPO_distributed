

.PHONY: train
train:
	torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost --nnodes=1 --nproc_per_node=4 train.py


