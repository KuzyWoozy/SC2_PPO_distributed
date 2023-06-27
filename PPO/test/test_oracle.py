import random
import torch as t
import numpy as np

from absl import app

from src.Parallel import SerialSGD
from src.starcraft.Agent import MiniStarAgent
from src.starcraft.Environment import StarcraftMinigame
from src.rl.Approximator import AtariNet
from src.rl.Loop import train_loop

from test.oracle.Parallel import SerialSGD as SerialSGD_oracle
from test.oracle.starcraft.Agent import MiniStarAgent as MiniStarAgent_oracle
from test.oracle.starcraft.Environment import StarcraftMinigame as StarcraftMinigame_oracle
from test.oracle.rl.Approximator import AtariNet as AtariNet_oracle
from test.oracle.rl.Loop import train_loop as train_loop_oracle


SEED = 0


def reset_seed():
    random.seed(SEED)
    t.manual_seed(SEED)
    np.random.seed(SEED)

def assert_models(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert t.equal(param1, param2)



def test_oracle():
   
    device = t.device("cpu")

    reset_seed()
    target = SerialSGD(AtariNet(), device) 
    
    def train_target(argv):                
        # Choose agent
        agent = MiniStarAgent(target)
        # Choose environment
        environment = StarcraftMinigame(agent)
        train_loop(agent, environment)

    try:
        app.run(train_target)
    except SystemExit:
        pass

    reset_seed()
    orac = SerialSGD_oracle(AtariNet_oracle(), device) 
    
    def train_oracle(argv):
        # Choose agent
        agent = MiniStarAgent_oracle(orac)
        # Choose environment
        environment = StarcraftMinigame_oracle(agent)
        train_loop_oracle(agent, environment)

    # Begin the training process
    try:
        app.run(train_oracle)
    except SystemExit:
        pass
    

    assert_models(target, orac)
