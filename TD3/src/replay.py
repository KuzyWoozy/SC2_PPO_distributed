import torch as t


class Batch:
    def __init__(self, states_t : t.Tensor, actions : t.Tensor, rewards : t.Tensor, 
            states_tt : t.Tensor, terminals : t.Tensor, n_gamma : t.Tensor) -> None:
        self._states_t = states_t
        self._actions = actions
        self._rewards = rewards
        self._states_tt = states_tt
        self._terminals = terminals
        self._n_gamma = n_gamma


    def get_states_t(self) -> t.Tensor:
        return self._states_t


    def get_actions(self) -> t.Tensor:
        return self._actions


    def get_rewards(self) -> t.Tensor:
        return self._rewards


    def get_states_tt(self) -> t.Tensor:
        return self._states_tt


    def get_terminals(self) -> t.Tensor:
        return self._terminals


    def get_n_gammas(self) -> t.Tensor:
        return self._n_gamma



class UniformReplay:
    def __init__(self, frame_skip : int, state_size : list[int], action_size : int,
            buffer_size : int) -> None:
        self._states_t = t.zeros(buffer_size, frame_skip + 1, *state_size)
        self._actions = t.zeros(buffer_size, action_size)
        self._rewards = t.zeros(buffer_size, 1)
        self._states_tt = t.zeros(buffer_size, frame_skip + 1, *state_size)
        self._terminals = t.zeros(buffer_size, 1)
        self._n_gamma = t.zeros(buffer_size, 1)

        self.i = 0
        self.size = 0
        self.buffer_size = buffer_size


    def store(self, state_t : t.Tensor, action : t.Tensor, reward : float, 
              state_tt : t.Tensor, terminal : bool, n_gamma : int) -> None:
        self._states_t[self.i] = state_t.clone()
        self._actions[self.i] = action
        self._rewards[self.i] = reward
        self._states_tt[self.i] = state_tt.clone()
        self._terminals[self.i] = terminal
        self._n_gamma[self.i] = n_gamma

        self.i = (self.i + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)


    def sample(self, batch_size : int = 100) -> Batch:
        samp_indx = t.randint(0, self.size, (batch_size,)) # Unif sampling

        states_t = self._states_t[samp_indx]
        actions = self._actions[samp_indx]
        rewards = self._rewards[samp_indx]
        states_tt = self._states_tt[samp_indx]
        terminals = self._terminals[samp_indx]
        n_gamma = self._n_gamma[samp_indx]

        return Batch(states_t, actions, rewards, states_tt, terminals, n_gamma)
