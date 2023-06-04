from pysc2.agents.random_agent import RandomAgent
from pysc2.env import sc2_env
from absl import app
import torch as t

import torch.distributed as dist


from src.rl.Loop import evaluate_loop
from src.rl.Approximator import MiniStarPolicy
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame
from src.Config import RUNMODE
from src.Parallel import DistSyncSGD



def main(argv):
 
    # Initialize distributed module if necessary
    if RUNMODE != "SERIAL":
        dist.init_process_group(backend="gloo")

    policy = MiniStarPolicy("checkpoints/FindAndDefeatZerglings-1100.chkpt")
    
    # Apply a parallel enabling wrapper to policy
    if RUNMODE == "DIST_SYNC":
        policy = DistSyncSGD(policy)

    # Choose agent
    agent = MiniStarAgent(policy)

    # Choose environment
    environment = StarcraftMinigame(agent, viz = True)
    
    # Begin the training process
    evaluate_loop(agent, environment)
   

if __name__ == "__main__":
    app.run(main)
