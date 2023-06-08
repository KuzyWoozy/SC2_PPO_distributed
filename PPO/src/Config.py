import torch as t

DEBUG = False

# Hyperparam
LEARNING_RATE = 3e-5

EPOCH_BATCH = 3 # MUST BE GREATER THAN 1 FOR CORRECTNESS (doesn't rly make sense for it to be <= 1 given PPO)

LAMBDA = 0.99
ENTROPY = 0.001

NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
CHECK_LOAD = None
CHECK_INTERVAL = 1_000

# Environment
MINIGAME_NAME = "FindAndDefeatZerglings"
MAX_AGENT_STEPS = 5_000_000

DTYPE = t.float32

# Distributed
ROOT = 0
PROCS_PER_NODE = 4
SYNC = False
GPU = True

