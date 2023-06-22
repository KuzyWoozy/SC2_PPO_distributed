from absl import app
import torch as t


from src.rl.Loop import evaluate_loop
from src.rl.Approximator import MiniStarPolicy
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame
from src.Parallel import SerialSGD

from src.Config import CHECK_LOAD



def main(argv):
    
    policy = MiniStarPolicy()
    
    if CHECK_LOAD:
        policy.load_state_dict(t.load(CHECK_LOAD, map_location = t.device("cpu"))["policy"])

    for para in policy.parameters():
        print(para)

    policy = SerialSGD(policy, t.device("cpu"))

    # Choose agent
    agent = MiniStarAgent(policy)

    # Choose environment
    environment = StarcraftMinigame(agent, viz = True)
    
    # Begin the training process
    evaluate_loop(agent, environment)
   

if __name__ == "__main__":
    app.run(main)
