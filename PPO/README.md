# PPO on StarCraft II LE
Implementation of the Proximal Policy Optimization Reinforcement Learning algorithm, uses DeepMind's StarCraft II for its learning environment for the variety of mini-games that it provides.
  
## Linux installation:
1. Install/Load Python 3.9 
	* module load python/3.9.13 on Cirrus
	* module load cray-python/3.9.13.1 on ARCHER2
	* 
2. Create a Python virtual environment via 'venv' and activate it
	* On ARCHER2/Cirrus unload the Python module before proceeding with installation of packages

3. Make sure all bash scripts in 'scripts/' have sufficient permissions to execute.

4. make install
	* By typing in the password ‘iagreetotheeula’ you agree to be bound by the terms of the Blizzard's [AI and Machine Learning License](http://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html)
	* Note that 'pip' may throw a recommendation to update warning, however this should be ignored as the installation script downgrades pip to satisfy specific dependencies. Note that by running this command, 

# Evaluating a model locally
1. (Optional) Select model in src/Config.py using the `CHECK_LOAD` parameter.
2. make eval

# Training on ARCHER2
1. (Optional) Modify 'src/Config.py' to adjust the number of nodes, parallel agents, hyperpameters etc.
2. Modify 'scripts/ARCHER2.slurm' to update its `source` command with your virtual environment 'your_virtual_env/bin/activate'.
3. make train_ARCHER2
4. Saved models will be periodically saved in 'checkpoints/'

# Training on Cirrus GPU partition
1. (Optional) Modify 'src/Config.py' to adjust the number of nodes, parallel agents, hyperpameters etc.
2.  Modify 'scripts/Cirrus_gpu.slurm' to update its `source` command with your virtual environment 'your_virtual_env/bin/activate'.
3. make train_Cirrus_gpu
4. Saved models will be periodically saved in 'checkpoints/'

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
	* Config.py
		- Central file for project configuration, should allow to modify any desired settings.
	* Misc.py
		- Miscellaneous and helpers functions.
	* Parallel.py
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


