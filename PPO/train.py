from pysc2.agents.random_agent import RandomAgent
from pysc2.env import sc2_env
from absl import app

import torch as t
import torch.distributed as dist


from src.rl.Loop import train_loop
from src.rl.Approximator import MiniStarPolicy
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame
from src.Parallel import DistSyncSGD, SerialSGD
from src.Config import SYNC, GPU, DTYPE, PROCS_PER_NODE, CHECK_LOAD


t.set_default_dtype(DTYPE)

if GPU:
    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True
    t.backends.cudnn.benchmark = True


def main(argv):
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
    policy = MiniStarPolicy(device)
        
    if CHECK_LOAD:
        policy.load_state_dict(t.load(CHECK_LOAD)["policy"])

    # Apply a parallel enabling wrapper to policy
    if SYNC:
        policy = DistSyncSGD(policy)
    else:
        policy = SerialSGD(policy)

    # Choose agent
    agent = MiniStarAgent(policy)
    # Choose environment
    environment = StarcraftMinigame(agent)
    
    # Begin the training process
    train_loop(agent, environment)


if __name__ == "__main__":
    app.run(main)
