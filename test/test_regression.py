import random, sys
import torch as t
import numpy as np

import test.oracle as oracle
import src

from train import (
    ENV,
    LEARN_RATE,
    FRAME_SKIP,
    HIDDEN_SIZE,
    REPLAY_BUFFER_SIZE,
    MINI_BATCH,
    TAU,
    NOISE_STD,
    CHECKPOINT_RATE,
    RaceEnvi
)

t.use_deterministic_algorithms(True)


SEED = 0

N_STEPS = 10
ENV_INTERACTIONS = 300
MAX_STEPS = 100
EXPLORE_STEPS = 30
REPLAY_BUFFER_SIZE = ENV_INTERACTIONS
CHECKPOINT_DIR = None

PERF_MEASURE = 100


def reset_randomness() -> None:
    random.seed(SEED)
    t.manual_seed(SEED)
    np.random.seed(SEED)

def evaluate_oracle() -> t.nn.Module:
    reset_randomness()

    env = RaceEnvi(FRAME_SKIP, seed = SEED)

    model = oracle.net.ActorCriticTwin(FRAME_SKIP, HIDDEN_SIZE, env.action_elems())

    trainer = oracle.trainer.T3D(
            lr = LEARN_RATE,
            act_delay_update = 2,
            a_size = env.action_elems(),
            replay_size = REPLAY_BUFFER_SIZE,
            max_steps = MAX_STEPS, explore_steps = EXPLORE_STEPS,
            n_steps = N_STEPS, frame_skip = FRAME_SKIP,
            batch_size = MINI_BATCH, tau = TAU,
            chk = oracle.misc.CheckpointManager(CHECKPOINT_DIR, ENV, CHECKPOINT_RATE),
            noise = oracle.misc.GaussianNoise([NOISE_STD, NOISE_STD, NOISE_STD]),
            perf = oracle.misc.PerformanceTracker(max_steps = MAX_STEPS, frame_skip = FRAME_SKIP, rate = PERF_MEASURE))
    
    trainer.train(ENV_INTERACTIONS, env, model)
    
    return model

def evaluate_sinner() -> t.nn.Module:
    reset_randomness()

    env = RaceEnvi(FRAME_SKIP, seed = SEED)

    model = src.net.ActorCriticTwin(FRAME_SKIP, HIDDEN_SIZE, env.action_elems())

    trainer = src.trainer.T3D(
            lr = LEARN_RATE,
            act_delay_update = 2,
            a_size = env.action_elems(),
            replay_size = REPLAY_BUFFER_SIZE,
            max_steps = MAX_STEPS, explore_steps = EXPLORE_STEPS,
            n_steps = N_STEPS, frame_skip = FRAME_SKIP,
            batch_size = MINI_BATCH, tau = TAU,
            chk = src.misc.CheckpointManager(CHECKPOINT_DIR, ENV, CHECKPOINT_RATE),
            noise = src.misc.GaussianNoise([NOISE_STD, NOISE_STD, NOISE_STD]),
            perf = src.misc.PerformanceTracker(max_steps = MAX_STEPS, frame_skip = FRAME_SKIP, rate = PERF_MEASURE))
    
    trainer.train(ENV_INTERACTIONS, env, model)

    return model

def assert_models(model1 : t.nn.Module, model2 : t.nn.Module) -> None:
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert t.equal(param1, param2)

def test_T3D() -> None:
    
    model_oracle = evaluate_oracle()
    model_sinner = evaluate_sinner()

    assert_models(model_oracle, model_sinner)
