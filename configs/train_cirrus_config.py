import torch as t
import os


DEBUG = False

ATARI_NET = False

# Hyperparam
LEARNING_RATE = 1e-4
GAMMA_DECAY = 0.99996


GAMMA = 0.99
ENTROPY = 3e-3
VALUE_COEFF = 1.0

NUM_ACTIONS = 7
NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
CHECK_LOAD = None
CHECK_INTERVAL = 10_000

# Environment
MINIGAME_NAME = "DefeatZerglingsAndBanelings"
TIMING_EPISODE_DELAY = 2
TRAJ = 30

# Distributed
PROFILE = False
ROOT = 0

REWARD_DEATH_SCALE = 2

MAX_NETWORK_UPDATES = None
MAX_TIME = 24
TIMER_INTERVAL = 1

if MAX_TIME is not None:
    MAX_TIME *= 60 * 60
    TIMER_INTERVAL *= 60 * 60


# Params for regression test
SEED = None
SYNC = False
GPU = True
CUDA_GRAPHS = False
AMP = False
COMPILE = False
MAX_AGENT_STEPS = None



if SYNC:
    PROCS_PER_NODE = int(os.environ["LOCAL_WORLD_SIZE"])
    PROCS = int(os.environ["WORLD_SIZE"])
else:
    PROCS_PER_NODE = 1
    PROCS = 1

CUDA_GRAPHS = GPU and CUDA_GRAPHS
