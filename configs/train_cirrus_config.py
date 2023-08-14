import torch as t
import os

# Turns on/off debug mode
DEBUG = False

# True for Atari-net, False for FullyConv
ATARI_NET = False

# Hyperparams
LEARNING_RATE = 1e-4
# Learning rate decay
LR_DECAY = 0.99996

# Return discount factor
GAMMA = 0.99
# Exploration rate
ENTROPY = 3e-3
# Importance coefficient of state-value approximation in loss
VALUE_COEFF = 1.0
# How much a policy update can deviate from the previous update iteration
PPO_CLIP = 0.2
# Lookahead TD-n number
TRAJ = 30

# Number of actions supported by the model
NUM_ACTIONS = 7
# Number of neurones in the model's dense layers
NN_HIDDEN_LAYER = 256


# Model to load before training/evaluation
CHECK_LOAD = None
# Save the model every # environment steps
CHECK_INTERVAL = 10_000

# Environment/minigame to train/evaluate
MINIGAME_NAME = "DefeatZerglingsAndBanelings"
# How many episodes to skip before timing starts (useful for JIT compilation to take place before timing)
TIMING_EPISODE_DELAY = 2

# Enable collective operations to keep track of correct metadata (for example, number of steps across ALL agents)
PROFILE = False
# Which process to be designated as root
ROOT = 0

# How much to scale marine death penalty in environments where they can be killed
REWARD_DEATH_SCALE = 2

# Maximum number of network updates to do
MAX_NETWORK_UPDATES = None
# Number of minutes to train for
MAX_TIME = 24
# Save model every # minutes
TIMER_INTERVAL = 1

if MAX_TIME is not None:
    MAX_TIME *= 60 * 60
    TIMER_INTERVAL *= 60 * 60


# Pseudorandom generator seed
SEED = None
# Whenever model will be used in a distributed setting
SYNC = True
# Whenever model will make use of a GPU (in serial or distributed setting)
GPU = True

# These options are not supported by Cirrus hardware CUDA version and should be ignored
CUDA_GRAPHS = False
AMP = False
COMPILE = False

# Maximum number of steps to take for an agent (enable PROFILE if want across ALL agents)
MAX_AGENT_STEPS = None


# Automatic variables, please ignore
if SYNC:
    PROCS_PER_NODE = int(os.environ["LOCAL_WORLD_SIZE"])
    PROCS = int(os.environ["WORLD_SIZE"])
else:
    PROCS_PER_NODE = 1
    PROCS = 1

CUDA_GRAPHS = GPU and CUDA_GRAPHS
