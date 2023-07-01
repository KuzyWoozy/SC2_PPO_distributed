import torch as t

DEBUG = False

ATARI_NET = True

# Hyperparam
LEARNING_RATE = 1e-4
GAMMA_DECAY = 0.99995


GAMMA = 0.99
ENTROPY = 1e-3
VALUE_COEFF = 1.0

NUM_ACTIONS = 7
NN_HIDDEN_LAYER = 256

PPO_CLIP = 0.2

# Checkpointing
CHECK_LOAD = "checkpoints/DefeatZerglingsAndBanelings-250000.chkpt"
#CHECK_LOAD = None
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
SEED = None
SYNC = True
GPU = False
MAX_AGENT_STEPS = 10_000_000
