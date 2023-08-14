

.PHONY: evaluate
evaluate:
	. sc_virt/bin/activate && cp configs/evaluate_config.py src/Config.py && SC2PATH="$(PWD)/SC2.4.9.3/StarCraftII" python evaluate.py

.PHONY: train_local
train_local:
	. sc_virt/bin/activate && cp configs/train_local_config.py src/Config.py && SC2PATH="$(PWD)/SC2.4.9.3/StarCraftII" torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost --nnodes=1 --nproc_per_node=4 train.py	

.PHONY: train_ARCHER2
train_ARCHER2:
	sbatch scripts/ARCHER2.slurm

.PHONY: train_cirrus
train_cirrus:
	sbatch scripts/Cirrus_cpu.slurm


.PHONY: regression_test
regression_test:
	. sc_virt/bin/activate && cp configs/regression_config.py src/Config.py && SC2PATH="$(PWD)/SC2.4.9.3/StarCraftII" python -m pytest

.PHONY: install_archer2
install_archer2:	
	 chmod a+x scripts/* && ./scripts/install_archer2.sh

.PHONY: install_cirrus
install_cirrus:	
	 chmod a+x scripts/* && ./scripts/install_cirrus.sh

.PHONY: install_local
install_local:
	chmod a+x scripts/* && ./scripts/install_local.sh
