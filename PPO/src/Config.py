import torch as t

DEBUG = False

ATARI_NET = True

# Hyperparam
LEARNING_RATE = 3e-5

EPOCH_BATCH = 3 # MUST BE GREATER THAN 1 FOR CORRECTNESS (doesn't rly make sense for it to be <= 1 given PPO)

GAMMA = 0.99
ENTROPY = 1e-2

NUM_ACTIONS = 12
NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
CHECK_LOAD = "checkpoints/DefeatRoaches-2120000.chkpt"
#CHECK_LOAD = None
CHECK_INTERVAL = 10_000

# Environment
SEED = None
MINIGAME_NAME = "DefeatRoaches"
MAX_AGENT_STEPS = 40_000
TIMING_EPISODE_DELAY = 2
TRAJ = 40
DTYPE = t.float32

# Distributed
PROFILE = True
ROOT = 0
PROCS_PER_NODE = 3
SYNC = True
GPU = False
