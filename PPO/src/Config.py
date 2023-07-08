import torch as t
import os


DEBUG = False

ATARI_NET = True

# Hyperparam
LEARNING_RATE = 1e-4
GAMMA_DECAY = 0.9999


GAMMA = 0.99
ENTROPY = 1e-5
VALUE_COEFF = 1.0

NUM_ACTIONS = 7
NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
CHECK_LOAD = "checkpoints/DefeatZerglingsAndBanelings-2390000.chkpt"
#CHECK_LOAD = None
CHECK_INTERVAL = 10_000

# Environment
MINIGAME_NAME = "DefeatZerglingsAndBanelings"
TIMING_EPISODE_DELAY = 2
TRAJ = 10
DTYPE = t.float32

# Distributed
PROFILE = False
ROOT = 0


MAX_NETWORK_UPDATES = None
MAX_TIME = None

if MAX_TIME is not None:
    MAX_TIME *= 60


# Params for regression test
SEED = 0
SYNC = False
GPU = False
MAX_AGENT_STEPS = 1_500



if SYNC:
    PROCS_PER_NODE = int(os.environ["LOCAL_WORLD_SIZE"])
    PROCS = int(os.environ["WORLD_SIZE"])
else:
    PROCS_PER_NODE = 1
    PROCS = 1
