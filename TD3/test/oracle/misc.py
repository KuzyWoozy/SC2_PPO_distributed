import copy
from typing import Any, Union, Optional
from io import TextIOWrapper

import torch as t
from src.envi import Envi


class GaussianNoise:
    def __init__(self, noise_std : list[float]) -> None:
        self.noise_dist = t.distributions.normal.Normal(0, t.tensor(noise_std))


    def gen(self, batch_size : int) -> Any:
        return self.noise_dist.sample([batch_size])



class CheckpointManager:
    @staticmethod
    def load(path : str, **kwargs : Union[t.nn.Module, t.optim.Adam])\
            -> dict[str, Union[t.nn.Module, t.optim.Adam]]:
        checkpoint = t.load(path)

        for key in kwargs:
            kwargs[key].load_state_dict(checkpoint[key])

        return kwargs


    def __init__(self, directory : Optional[str], name : str, rate : int) -> None:
        self.directory = directory
        self.name = name
        self.rate = rate
        self.save_iter = rate


    def time_to_save(self, step : int) -> bool:
        if step >= self.save_iter:
            return True
        return False


    def save(self, step : int, **kwargs : Union[t.nn.Module, t.optim.Adam]) -> None:
        self.save_iter += self.rate
        if not self.directory is None:
            t.save({key : kwargs[key].state_dict() for key in kwargs},\
               self.directory + "/" + self.name + f"-{step}.chkpt")



class PerformanceTracker:
    def __init__(self, max_steps : int, frame_skip : int, rate : int,
            file : Optional[TextIOWrapper] = None) -> None:
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.rate = rate
        self.measure_iter = rate

        self.trajectories = 3
        self.file = file


    def evaluate(self, env : Envi, model : t.nn.Module) -> float:
        model.eval()

        earned = 0

        with t.no_grad():
            for _ in range(self.trajectories):
                state_tt, _ = env.reset()

                for _ in range(self.max_steps):
                    state_t = state_tt

                    action = model(state_t.unsqueeze(0)).detach()

                    state_t = t.roll(state_t, shifts = -self.frame_skip, dims = 0)
                    for skip in range(self.frame_skip):
                        state, reward, terminal, _, _ = env.step(action)
                        state_t[skip + 1] = state
                        earned += reward

                        if terminal:
                            break

                    if terminal:
                        break

                    state, reward, terminal, _, _ = env.step(action)
                    state_tt = t.roll(state_t, shifts = -1, dims = 0)
                    state_tt[self.frame_skip] = state
                    earned += reward

                    if terminal:
                        break

        model.train()

        return earned / self.trajectories


    def save(self, steps : int, reward : float) -> None:
        if self.file:
            self.file.write(f"{steps}, {reward}\n")
            self.file.flush()


    def time_to_measure(self, steps : int) -> bool:
        return steps >= self.measure_iter


    def measure(self, steps : int, env : Envi, model : t.nn.Module) -> None:
        self.save(steps, self.evaluate(copy.deepcopy(env), model))
        self.measure_iter += self.rate
