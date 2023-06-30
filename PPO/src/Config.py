import torch as t

DEBUG = False

ATARI_NET = True

# Hyperparam
LEARNING_RATE = 3e-4

GAMMA = 0.99
ENTROPY = 1e-5
VALUE_COEFF = 1.0

NUM_ACTIONS = 7
NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
CHECK_LOAD = "checkpoints/MoveToBeacon-70000.chkpt"
#CHECK_LOAD = None
CHECK_INTERVAL = 10_000

# Environment
MINIGAME_NAME = "MoveToBeacon"
TIMING_EPISODE_DELAY = 2
TRAJ = 6
DTYPE = t.float32

# Distributed
PROFILE = False
ROOT = 0
PROCS_PER_NODE = 4


# Params for regression test
SEED = None
SYNC = True
GPU = False
MAX_AGENT_STEPS = 10_000_000
