import torch as t


def categorical_sample(probs):
    return t.distributions.Categorical(probs = probs).sample((1,)).item()

