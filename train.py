import gym, os
import torch as t

from src.net import ActorCriticTwin
from src.trainer import T3D
from src.envi import Envi
from src.misc import CheckpointManager, GaussianNoise, PerformanceTracker

from typing import Any, Optional


# SETTINGS
ENV = "CarRacing-v2"

class RaceEnvi(Envi):
    def __init__(self, frame_skip : int, render_mode : Optional[str]= None, seed : Optional[int] = None) -> None:
        env = gym.make(ENV, render_mode = render_mode)
        super().__init__(env, frame_skip, seed)
        
    def preprocess(self, x : t.Tensor) -> t.Tensor:
        x = x[:-12, 6:-6, :]
        x = x @ t.tensor([0.3, 0.59, 0.11]) # greyscale
        x /= 255.0
        
        #x = x.squeeze() 
        #plt.imshow(x)
        #plt.show(block = False)
        #plt.pause(0.001) 
        
        return x

    def obs_shape(self) -> list[int]:
        return [84, 84]

    def action_elems(self) -> int:
        return int(self.env.action_space.shape[0])


LEARN_RATE = 1e-5
ENV_INTERACTIONS = 1_000_000
MAX_STEPS = 1_000
EXPLORE_STEPS = 20_000

N_STEPS = 20
FRAME_SKIP = 3

REPLAY_BUFFER_SIZE = 30_000
MINI_BATCH = 20

TAU = 0.005
NOISE_STD = 0.1
HIDDEN_SIZE = 256

CHECKPOINT_RATE = ENV_INTERACTIONS // 20
CHECKPOINT_DIR = "checkpoints"

PERF_MEASURE = 10_000
PERF_FILE = open("perf.csv", "w")


# MAIN
def main() -> None:
    
    if (not os.path.isdir(CHECKPOINT_DIR)):
        os.mkdir(CHECKPOINT_DIR)

    #dev = selectDevice(override=)
    env = RaceEnvi(FRAME_SKIP)

    model = ActorCriticTwin(FRAME_SKIP, HIDDEN_SIZE, env.action_elems())

    trainer = T3D(
            lr = LEARN_RATE,
            act_delay_update = 2,
            a_size = env.action_elems(),
            replay_size = REPLAY_BUFFER_SIZE,
            max_steps = MAX_STEPS, explore_steps = EXPLORE_STEPS,
            n_steps = N_STEPS, frame_skip = FRAME_SKIP,
            batch_size = MINI_BATCH, tau = TAU,
            chk = CheckpointManager(CHECKPOINT_DIR, ENV, CHECKPOINT_RATE),
            noise = GaussianNoise([NOISE_STD, NOISE_STD, NOISE_STD]),
            perf = PerformanceTracker(max_steps = MAX_STEPS, frame_skip = FRAME_SKIP, rate = PERF_MEASURE, file = PERF_FILE))

    
    trainer.train(ENV_INTERACTIONS, env, model)

    PERF_FILE.close()

if __name__ == "__main__":
    main()
