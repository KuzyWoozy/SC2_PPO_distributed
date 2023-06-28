from absl import app
import torch as t


from src.rl.Loop import evaluate_loop
from src.rl.Approximator import AtariNet, FullyConv
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame
from src.Parallel import SerialSGD

from src.Config import CHECK_LOAD, ATARI_NET



def main(argv):
    
    if ATARI_NET:
        policy = AtariNet()
    else:
        policy = FullyConv()
    

    if CHECK_LOAD:
        policy.load_state_dict(t.load(CHECK_LOAD, map_location = t.device("cpu"))["policy"])

    policy = SerialSGD(policy, t.device("cpu"))

    # Choose agent
    agent = MiniStarAgent(policy)
    

    # Choose environment
    environment = StarcraftMinigame(agent, viz = False)
    
    # Begin the training process
    print("Evaluation score:", evaluate_loop(agent, environment, 100))
   

if __name__ == "__main__":
    app.run(main)
