from absl import app

import random
import numpy as np
import torch as t
import torch.distributed as dist

from src.rl.Loop import train_loop
from src.rl.Approximator import MiniStarPolicy
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame
from src.Parallel import DistSyncSGD, SerialSGD
from src.Config import SYNC, GPU, DTYPE, PROCS_PER_NODE, CHECK_LOAD, DEBUG, SEED
from src.Misc import module_params_count, verify_config


if SEED is not None: 
    random.seed(SEED)
    t.manual_seed(SEED)
    np.random.seed(SEED)


if GPU:
    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True
    t.backends.cudnn.benchmark = False if SEED else True

if DEBUG:
    t.autograd.set_detect_anomaly(True, check_nan=True)
    if GPU:
        t.cuda.set_sync_debug_mode(1) # Objectively best flag to ever be implemented in a library

t.set_default_dtype(DTYPE)


def main(argv):
    
    verify_config()

    # Initialize distributed module if necessary
    if SYNC:
        if GPU:
            dist.init_process_group(backend="nccl")
            device = t.device("cuda", dist.get_rank() % PROCS_PER_NODE) 
        else:
            dist.init_process_group(backend="gloo")
            device = t.device("cpu")
    else:
        if GPU:
            device = t.device("cuda")
        else:
            device = t.device("cpu")
    

    # Choose policy
    policy = MiniStarPolicy()
        
    if CHECK_LOAD:
        policy.load_state_dict(t.load(CHECK_LOAD)["policy"])

    

    # Apply a parallel enabling wrapper to policy
    if SYNC:
        policy = DistSyncSGD(policy, device)
    else:
        policy = SerialSGD(policy, device)
    

    print("Model parameter count:", module_params_count(policy))

    # Choose agent
    agent = MiniStarAgent(policy)
    # Choose environment
    environment = StarcraftMinigame(agent)

    
    # Begin the training process
    train_loop(agent, environment)


if __name__ == "__main__":
    app.run(main)
