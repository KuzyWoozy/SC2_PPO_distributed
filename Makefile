

.PHONY: eval
eval:
	. sc_virt/bin/activate && SC2PATH="$(PWD)/SC2.4.9.3/StarCraftII" python evaluate.py

.PHONY: train
train:
	. sc_virt/bin/activate && SC2PATH="$(PWD)/SC2.4.9.3/StarCraftII" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost --nnodes=1 --nproc_per_node=4 train.py	

.PHONY: train_archer2
train_ARCHER2:
	sbatch scripts/ARCHER2.slurm

.PHONY: train_cirrus_cpu
train_cirrus_cpu:
	sbatch scripts/Cirrus_cpu.slurm


.PHONY: test
test:
	SC2PATH="$(PWD)/SC2.4.9.3/StarCraftII" python -m pytest

.PHONY: install_archer2
install_archer2:	
	 chmod a+x scripts/* && ./scripts/install_archer2.sh

.PHONY: install_cirrus
install_cirrus:	
	 chmod a+x scripts/* && ./scripts/install_cirrus.sh


