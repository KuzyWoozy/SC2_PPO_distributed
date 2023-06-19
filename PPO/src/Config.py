import torch as t

DEBUG = False

# Hyperparam
LEARNING_RATE = 1e-4

EPOCH_BATCH = 3 # MUST BE GREATER THAN 1 FOR CORRECTNESS (doesn't rly make sense for it to be <= 1 given PPO)

LAMBDA = 0.99
ENTROPY = 0.001

NUM_ACTIONS = 12
NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
#CHECK_LOAD = "checkpoints/FindAndDefeatZerglings-10000.chkpt"
CHECK_LOAD = None
CHECK_INTERVAL = 10_000

# Environment
SEED = None
MINIGAME_NAME = "FindAndDefeatZerglings"
MAX_AGENT_STEPS = 3_000_000
TIMING_EPISODE_DELAY = 2
TRAJ = 30
DTYPE = t.float32

# Distributed
ROOT = 0
PROCS_PER_NODE = 128
SYNC = False
GPU = True
