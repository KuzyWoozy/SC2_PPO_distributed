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
from src.Config import SYNC, GPU, DTYPE



if GPU:
    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True


def main(argv):
    # Initialize distributed module if necessary
    if SYNC:
        if GPU:
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")

    if GPU:
        device = t.device("cuda") # Only works single node, single-gpu 
    else:
        device = t.device("cpu")

    # Choose policy
    policy = MiniStarPolicy()
        
    # Apply a parallel enabling wrapper to policy
    if SYNC:
        policy = DistSyncSGD(policy, device = device)
    else:
        policy = SerialSGD(policy, device = device)

    # Choose agent
    agent = MiniStarAgent(policy)
    # Choose environment
    environment = StarcraftMinigame(agent)
    
    # Begin the training process
    train_loop(agent, environment)


if __name__ == "__main__":
    app.run(main)
