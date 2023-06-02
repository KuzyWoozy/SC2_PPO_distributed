from pysc2.agents.random_agent import RandomAgent
from pysc2.env import sc2_env
from absl import app

import torch as t
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from src.Environment import FDZ
from src.Agent import *
from src.Checkpoint import CheckpointManager
from src.Loop import run_train_loop
from src.Config import DEBUG_MODE, MAX_EPISODES, CHECK_INTERVAL
from src.Approximator import FDZApprox, DistributedAgentWorkaround


if DEBUG_MODE:
    t.autograd.set_detect_anomaly(True)


def main(argv):
    dist.init_process_group(backend="gloo")
    
    agent = FDZAgent(FDZApprox(), check_manager = CheckpointManager("checkpoints", "findAndDefeatZ", CHECK_INTERVAL))

    workaround = DDP(DistributedAgentWorkaround(agent), find_unused_parameters = True, gradient_as_bucket_view = True, broadcast_buffers = False)

    run_train_loop(workaround, agent, FDZ(agent), MAX_EPISODES)
        

if __name__ == "__main__":
    app.run(main)
