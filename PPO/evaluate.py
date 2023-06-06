from pysc2.agents.random_agent import RandomAgent
from pysc2.env import sc2_env
from absl import app
import torch as t


from src.rl.Loop import evaluate_loop
from src.rl.Approximator import MiniStarPolicy
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame



def main(argv):
 
    policy = MiniStarPolicy()
    
    # Choose agent
    agent = MiniStarAgent(policy, load_model = "checkpoints/FindAndDefeatZerglings-710000.chkpt")

    # Choose environment
    environment = StarcraftMinigame(agent, viz = True)
    
    # Begin the training process
    evaluate_loop(agent, environment)
   

if __name__ == "__main__":
    app.run(main)
