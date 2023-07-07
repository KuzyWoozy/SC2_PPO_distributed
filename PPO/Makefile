

.PHONY: eval
eval:
	SC2PATH="$PWD/SC2.4.9.3/StarCraftII" python evaluate.py

.PHONY: train
train:
	SC2PATH="$PWD/SC2.4.9.3/StarCraftII" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost --nnodes=1 --nproc_per_node=1 train.py

.PHONY: test
test:
	SC2PATH=$PWD/SC2.4.9.3/StarCraftII python -m pytest

.PHONY: install
install:	
	 ./scripts/install.sh


