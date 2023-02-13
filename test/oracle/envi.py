import abc
from typing import Any, Optional

import torch as t


class Envi(abc.ABC):
    def __init__(self, env : Any, frame_skip : int, seed : Optional[int]) -> None:
        super().__init__()
        self.env = env
        self.frame_skip = frame_skip
        self._steps_taken = 0

        if not seed is None:
            self.env.action_space.seed(seed)
            self.env.reset(seed = seed)


    def reset(self) -> tuple[t.Tensor, dict[str, Any]]:
        initial_state, info = self.env.reset(options={"randomize": False})
        initial_state = self.preprocess(t.from_numpy(initial_state).float())

        state = t.empty(self.frame_skip + 1, *initial_state.shape)
        state[:] = initial_state

        return state, info


    def step(self, action : t.Tensor) -> tuple[t.Tensor, int, bool, bool, dict[str, Any]]:
        state, reward, terminal, trunc, info = self.env.step(action.squeeze().numpy())
        state = self.preprocess(t.from_numpy(state).float())

        self._steps_taken += 1

        return state, reward, terminal, trunc, info


    def random_action(self) -> t.Tensor:
        return t.from_numpy(self.env.action_space.sample()).float()


    def reset_steps(self) -> None:
        self._steps_taken = 0


    def steps_taken(self) -> int:
        return self._steps_taken


    def max_action(self) -> t.Tensor:
        return t.from_numpy(self.env.action_space.high).float()


    def min_action(self) -> t.Tensor:
        return t.from_numpy(self.env.action_space.low).float()


    @abc.abstractmethod
    def preprocess(self, x : t.Tensor) -> t.Tensor:
        pass


    @abc.abstractmethod
    def obs_shape(self) -> list[int]:
        pass


    @abc.abstractmethod
    def action_elems(self) -> int:
        pass
