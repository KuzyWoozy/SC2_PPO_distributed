import random
import torch as t
import numpy as np

from absl import app

from src.Agent import FDZAgent
from src.Environment import FDZ
from src.Loop import run_train_loop
from src.Config import EPISODE_BATCH

from test.oracle.Agent import FDZAgent as FDZAgent_oracle
from test.oracle.Environment import FDZ as FDZ_oracle 
from test.oracle.Loop import run_train_loop as run_train_loop_oracle


SEED = 0
TEST_STEPS = 1500



def reset_seed():
    random.seed(SEED)
    t.manual_seed(SEED)
    np.random.seed(SEED)

def assert_models(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert t.equal(param1, param2)



def test_oracle():
    
    reset_seed()
    target = FDZAgent()

    def train_target(argv):                
        env = FDZ(target, seed = SEED)
        run_train_loop(target, env, episode_batch = EPISODE_BATCH, max_frames = TEST_STEPS)

    try:
        app.run(train_target)
    except SystemExit:
        pass
    

    reset_seed()
    orac = FDZAgent_oracle()
    
    def train_oracle(argv):                
        env = FDZ_oracle(orac, seed = SEED)
        run_train_loop_oracle(orac, env, episode_batch = EPISODE_BATCH, max_frames = TEST_STEPS)

    try:
        app.run(train_oracle)
    except SystemExit:
        pass
    

    assert_models(target.approx, orac.approx)
