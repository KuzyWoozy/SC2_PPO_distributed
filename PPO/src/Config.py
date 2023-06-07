import torch as t


# Hyperparam
LEARNING_RATE = 3e-5

EPOCH_BATCH = 3

LAMBDA = 0.99
ENTROPY = 0.001

NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
CHECK_INTERVAL = 10_000

# Environment
MINIGAME_NAME = "FindAndDefeatZerglings"
MAX_AGENT_STEPS = 5_000_000

# Distributed
ROOT = 0

SYNC = False
GPU = True

DTYPE = t.float32
