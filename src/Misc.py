import torch as t
import os

from src.Config import LEARNING_RATE, GAMMA, ENTROPY, NN_HIDDEN_LAYER, NUM_ACTIONS, PPO_CLIP, CHECK_INTERVAL, MAX_AGENT_STEPS, PROCS_PER_NODE



class CheckpointManager:
        
    def __init__(self, directory : str, name : str, rate : int = 10_000) -> None:
        """
        Responsible for storing models at a given rate within the specified directory. 

        Parameters
        ----------
        directory : str
            Directory to store saved models.
        name : str
            Model name.
        rate : int    
            Save every # steps.
        """

        self.directory = directory
        self.name = name
        self.rate = rate
        self.next_checkpoint = rate

        if not os.path.isdir(directory):
            os.mkdir(directory)


    def time_to_save(self, step : int) -> bool:
        """
        Check whenever it is time to save a model. 

        Parameters
        ----------
        step : int
            Current agent step.

        Returns
        -------
        out : bool
            True if time to save, False otherwise.
        """

        if step >= self.next_checkpoint:
            self.next_checkpoint += self.rate
            return True
        return False
    

    def save(self, step : int, state_dict) -> None:
        """
        Save the model state.

        Parameters
        ----------
        step : int
            Current step in training.
        state_dict : dict[str, t.nn.Model]
            Pytorch model dictionary.
        """

        t.save(state_dict, self.directory + "/" + self.name + f"-{step}.chkpt")
    

def categorical_sample(probs):
    """
    Perform a categorical sample from the specified log probabilities.

    Parameters
    ----------
    step : t.Tensor
        Log probabilities to sample from.

    Returns
    -------
    out : int
        Sampled categorical element.
    """
    
    return t.distributions.Categorical(logits = probs).sample((1,)).item()


def module_params_count(module):
    """
    Count the number of learnable parameters in the specified model.

    Parameters
    ----------
    module : t.nn.Module
        Model to count parameters from.

    Returns
    -------
    out : int
        Number of model parameters.
    """

    return sum([par.numel() for par in module.parameters()])

def verify_config():
    "Verify configuration is sensible."

    assert LEARNING_RATE < 0.1 and "Learning rate too high (>= 0.1)"
    assert GAMMA >= 0.0 and GAMMA <= 1.0 and "Lambda must be in range [0.0, 1.0]"
    assert ENTROPY < 1.0 and "Entropy too high (>= 1.0)"
    assert NN_HIDDEN_LAYER > 0 and "Hidden layer must be positive (> 0)"
    assert NUM_ACTIONS > 0 and "Number of actions must be positive (> 0)"
    assert PPO_CLIP > 0.0 and "Clip must be positive (> 0)"
    assert CHECK_INTERVAL > 0 and "Checkpoint interval must be positive (> 0)"
    assert PROCS_PER_NODE > 0 and "Processors per node must be positive (> 0)"

