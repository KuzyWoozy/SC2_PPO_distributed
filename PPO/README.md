# PPO on StarCraft II LE
Implementation of the *Proximal Policy Optimization* Reinforcement Learning algorithm, uses DeepMind's StarCraft II Learning Environment for the variety of mini-games that it provides.

[![](data/images/thumbnail.png)](https://www.youtube.com/embed/uk2abOIxBak)

## Linux installation:
The installation requires agreement to the terms of [BLIZZARD STARCRAFT II AI AND MACHINE LEARNING LICENSE](http://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html), by typing in the password '**iagreetotheeula**' during the installation process you agree to be bound by these terms.

<ins>**ARCHER2:**</ins>

* `make install_archer2`

<ins>**Cirrus:**</ins>

* `make install_cirrus`

Note that **pip** may throw a recommendation to update warning, however this should be <ins>ignored</ins> as the installation script downgrades pip to satisfy specific dependencies.

## Evaluating a model locally:
1. `(Optional) Select model in src/Config.py using the 'CHECK_LOAD' parameter.`
2. `make eval`

## Training on ARCHER2:
<img src="data/images/archer2_logo.png" alt="drawing" width="200"/>

1. `(Optional) Modify 'src/Config.py' to adjust hyperpameters, distributed/gpu training, policy model, pseudorandom seeds etc.`
2. `make train_archer2`
3. `Saved models will be periodically saved in 'checkpoints/'`

## Training on Cirrus:
<img src="data/images/cirrus_logo.png" alt="drawing" width="180"/>

1. `(Optional) Modify 'src/Config.py' to adjust hyperpameters, distributed/gpu training, policy model, pseudorandom seeds etc.`
2. <ins>**CPU:**</ins> `make train_cirrus_cpu`

3. `Saved models will be periodically saved in 'checkpoints/'`

## Running regression tests:
* `make test`
  
## Directories overview:
- `evaluate.py`
	* Evaluates a model checkpoint.
- `train.py`
	* Entry point for the training procedure.
- `checkpoints/`
	* Saved models or models to be evaluated location.
- `scripts/`
-	* SLURM job scripts for ARCHER2 and Cirrus work launching.
- `data/`
	* Data from experiment/SLURM runs.
- `src/`
	* `Config.py`
		- Central file for project configuration, should allow to modify any desired settings.
	* `Misc.py`
		- Miscellaneous and helpers functions.
	* `Parallel.py`
		- Responsible for providing parallel functionality wrappers to the agent policy and hence to be trained on multi-core/gpu systems.
	* `rl/`
		- `Approximator.py`
			* Atari-net and FullyConv agent policy implementations.
		- `Loop`
			* Training and evaluation loop implementations, responsible for tying together all the RL components.
	* `starcraft/`
		- `Agent.py`
			* Responsible for updating the agent policy by piping feedback from the training environment in the form of scalar rewards.
		- `Environment.py`
			* Setup for the StarCraft II environments, configuring the mini-game type, rules and feature/action space.
- `test/`
	* `oracle/`
		- Snapshot of the project implementation.
	* `test_oracle.py`
		- Regression testing framework evaluator.
- `Makefile`
	* Configuration file for `make` containing various helper routines.


