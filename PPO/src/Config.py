import torch as t

DEBUG = False

ATARI_NET = True

# Hyperparam
LEARNING_RATE = 1e-3

EPOCH_BATCH = 3 # MUST BE GREATER THAN 1 FOR CORRECTNESS (doesn't rly make sense for it to be <= 1 given PPO)

GAMMA = 0.99
ENTROPY = 1e-3
VALUE_COEFF = 0.5

NUM_ACTIONS = 7
NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
#CHECK_LOAD = "checkpoints/MoveToBeacon-300000.chkpt"
CHECK_LOAD = None
CHECK_INTERVAL = 10_000

# Environment
MINIGAME_NAME = "MoveToBeacon"
TIMING_EPISODE_DELAY = 2
TRAJ = 10
DTYPE = t.float32

# Distributed
PROFILE = False
ROOT = 0
PROCS_PER_NODE = 4


# Params for regression test
SEED = None
SYNC = False
GPU = False
MAX_AGENT_STEPS = 10_000_000
