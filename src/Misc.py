import torch as t
import os

from src.Config import LEARNING_RATE, GAMMA, ENTROPY, NN_HIDDEN_LAYER, NUM_ACTIONS, PPO_CLIP, CHECK_INTERVAL, MAX_AGENT_STEPS, PROCS_PER_NODE



class CheckpointManager:
        
    def __init__(self, directory : str, name : str, rate : int = 10_000) -> None:
        self.directory = directory
        self.name = name
        self.rate = rate
        self.next_checkpoint = rate

        if not os.path.isdir(directory):
            os.mkdir(directory)


    def time_to_save(self, step : int) -> bool:
        if step >= self.next_checkpoint:
            self.next_checkpoint += self.rate
            return True
        return False
    

    def save(self, step : int, state_dict) -> None:
        t.save(state_dict, self.directory + "/" + self.name + f"-{step}.chkpt")
    

def categorical_sample(probs):
    return t.distributions.Categorical(logits = probs).sample((1,)).item()


def module_params_count(module):
    return sum([par.numel() for par in module.parameters()])

def verify_config():
    assert LEARNING_RATE < 0.1 and "Learning rate too high (>= 0.1)"
    assert GAMMA >= 0.0 and GAMMA <= 1.0 and "Lambda must be in range [0.0, 1.0]"
    assert ENTROPY < 1.0 and "Entropy too high (>= 1.0)"
    assert NN_HIDDEN_LAYER > 0 and "Hidden layer must be positive (> 0)"
    assert NUM_ACTIONS > 0 and "Number of actions must be positive (> 0)"
    assert PPO_CLIP > 0.0 and "Clip must be positive (> 0)"
    assert CHECK_INTERVAL > 0 and "Checkpoint interval must be positive (> 0)"
    assert PROCS_PER_NODE > 0 and "Processors per node must be positive (> 0)"

