from pysc2.agents.random_agent import RandomAgent
from pysc2.env import sc2_env
from absl import app

import torch as t
import torch.distributed as dist


from src.rl.Loop import train_loop
from src.rl.Approximator import MiniStarPolicy
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame
from src.Parallel import DistSyncSGD
from src.Config import RUNMODE


def main(argv):
    # Initialize distributed module if necessary
    if RUNMODE != "SERIAL":
        dist.init_process_group(backend="gloo")
    
    # Choose policy
    policy = MiniStarPolicy()
   
    # Apply a parallel enabling wrapper to policy
    if RUNMODE == "DIST_SYNC":
        policy = DistSyncSGD(policy)

    # Choose agent
    agent = MiniStarAgent(policy)
    # Choose environment
    environment = StarcraftMinigame(agent)
    
    # Begin the training process
    train_loop(agent, environment)


if __name__ == "__main__":
    app.run(main)
