import torch as t

DEBUG = False

ATARI_NET = True

# Hyperparam
LEARNING_RATE = 3e-5

EPOCH_BATCH = 3 # MUST BE GREATER THAN 1 FOR CORRECTNESS (doesn't rly make sense for it to be <= 1 given PPO)

GAMMA = 0.99
ENTROPY = 1e-1

NUM_ACTIONS = 7
NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.1

# Checkpointing
# CHECK_LOAD = "checkpoints/DefeatZerglingsAndBanelings-990000.chkpt"
CHECK_LOAD = None
CHECK_INTERVAL = 10_000

# Environment
MINIGAME_NAME = "DefeatZerglingsAndBanelings"
TIMING_EPISODE_DELAY = 2
TRAJ = 40
DTYPE = t.float32

# Distributed
PROFILE = False
ROOT = 0
PROCS_PER_NODE = 128


# Params for regression test
SEED = 0
SYNC = False
GPU = False
MAX_AGENT_STEPS = 1_500
