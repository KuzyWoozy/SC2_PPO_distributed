from pysc2.agents.random_agent import RandomAgent
from pysc2.env import sc2_env
from absl import app
import torch as t


from src.rl.Loop import evaluate_loop
from src.rl.Approximator import MiniStarPolicy
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame

from src.Config import CHECK_LOAD



def main(argv):
    
    policy = MiniStarPolicy(t.device("cpu"))
    
    if CHECK_LOAD:
        policy.load_state_dict(t.load(CHECK_LOAD, map_location = t.device("cpu"))["policy"])

    # Choose agent
    agent = MiniStarAgent(policy)

    # Choose environment
    environment = StarcraftMinigame(agent, viz = True)
    
    # Begin the training process
    evaluate_loop(agent, environment)
   

if __name__ == "__main__":
    app.run(main)
