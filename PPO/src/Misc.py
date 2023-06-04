import torch as t
import os


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
    return t.distributions.Categorical(probs = probs).sample((1,)).item()

