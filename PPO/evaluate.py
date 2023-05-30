from pysc2.agents.random_agent import RandomAgent
from pysc2.env import sc2_env
from absl import app
import torch as t

from src.Environment import FDZ
from src.Agent import *
from src.Checkpoint import CheckpointManager
from src.Loop import run_evaluate_loop
from src.Config import DEBUG_MODE, EPISODE_BATCH, MAX_FRAMES



if DEBUG_MODE:
    t.autograd.set_detect_anomaly(True)


def main(argv):
    agent = FDZAgent(CheckpointManager("checkpoints", "findAndDefeatZ"))

    agent.check_manager.load(4_000_000, approx = agent.approx)

    run_evaluate_loop(agent, FDZ(agent, viz = True))
        

if __name__ == "__main__":
    app.run(main)
