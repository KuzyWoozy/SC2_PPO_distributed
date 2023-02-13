import copy
from typing import Optional

import torch as t

from src.replay import UniformReplay, Batch
from src.envi import Envi
from src.misc import CheckpointManager, GaussianNoise, PerformanceTracker


# Simple container for QoL
class ACContext:
    def __init__(self, model : t.nn.Module, model_target : t.nn.Module,
            model_opti : t.optim.Adam) -> None:
        self.model = model
        self.model_target = model_target
        self.model_opti = model_opti


    def get_model(self) -> t.nn.Module:
        return self.model


    def get_model_target(self) -> t.nn.Module:
        return self.model_target


    def get_model_opti(self) -> t.optim.Adam:
        return self.model_opti



class T3D:
    def __init__(self,
            lr : float,
            act_delay_update : int,
            a_size : int,
            replay_size : int, max_steps : int, explore_steps : int,
            n_steps : int, frame_skip : int,
            batch_size : int, tau : float,
            chk : CheckpointManager,
            noise : GaussianNoise,
            perf : PerformanceTracker) -> None:
        self.lr = lr
        self.act_delay_update = act_delay_update
        self.a_size = a_size

        self.replay_size = replay_size
        self.max_steps = max_steps
        self.explore_steps = explore_steps
        self.n_steps = n_steps
        self.frame_skip = frame_skip

        self.tau = tau
        self.batch_size = batch_size

        self.chk = chk
        self.noise = noise
        self.perf = perf

        self.update_counter = 0
        # Params taken from T3D paper
        self.target_noise = GaussianNoise([0.2, 0.1, 0.1])
        self.target_noise_cutoff = t.tensor([0.5, 0.25, 0.25])


    def _train_mode(self, *args : t.nn.Module) -> None:
        for arg in args:
            arg.train()


    def _eval_mode(self, *args : t.nn.Module) -> None:
        for arg in args:
            arg.eval()


    def _freeze_model(self, model : t.nn.Module) -> t.nn.Module:
        return copy.deepcopy(model).requires_grad_(False)


    def _sync_target_model(self, mdl : t.nn.Module, mdl_tar : t.nn.Module,
            tau : float = 0.001) -> None:
        for mdl_par, mdl_tar_par in zip(mdl.parameters(), mdl_tar.parameters()):
            mdl_tar_par.copy_(tau * mdl_par.detach() + (1 - tau) * mdl_tar_par)


    def _regularize_action(self, model : t.nn.Module, env : Envi,
            state : t.Tensor) -> t.Tensor:
        return t.clip(model(state) + t.clip(self.target_noise.gen(self.batch_size),
            -self.target_noise_cutoff, self.target_noise_cutoff), env.min_action(),
            env.max_action())


    def _q_target(self, env : Envi, ac_context : ACContext, batch : Batch,
            gamma : float = 0.99) -> t.Tensor:
        model_target = ac_context.get_model_target()

        rewards = batch.get_rewards()
        states_tt = batch.get_states_tt()
        terminals = batch.get_terminals()
        n_gammas = batch.get_n_gammas()

        action = self._regularize_action(model_target, env, states_tt)
        
        _, c_t, twin_c_t = model_target(states_tt, action)
        return rewards + (1.0 - terminals) * (gamma ** n_gammas) *\
            t.minimum(c_t, twin_c_t)


    def _update_critics(self, model : t.nn.Module, model_opti : t.optim.Adam,
            q_targets : t.Tensor, batch : Batch) -> None:
        states_t = batch.get_states_t()
        actions = batch.get_actions()

        loss = model.critics_loss_func(states_t, actions, q_targets)

        model_opti.zero_grad()
        loss.backward()
        model_opti.step()


    def _update_actor(self, model : t.nn.Module, model_opti : t.optim.Adam,
            batch : Batch) -> None:
        states_t = batch.get_states_t()

        loss = model.actor_loss_func(states_t)

        model_opti.zero_grad()
        loss.backward()
        model_opti.step()


    def _exploit_xp(self, env : Envi, replay : UniformReplay,
            ac_context : ACContext) -> None:
        model = ac_context.get_model()
        model_target = ac_context.get_model_target()
        model_opti = ac_context.get_model_opti()

        self._train_mode(model)

        batch = replay.sample(self.batch_size)

        q_targets = self._q_target(env, ac_context, batch)

        self._update_critics(model, model_opti, q_targets, batch)

        self.update_counter += 1
        if self.update_counter % self.act_delay_update == 0:
            self._update_actor(model, model_opti, batch)
            self._sync_target_model(mdl = model, mdl_tar = model_target, tau = self.tau)


    def _explore(self, env : Envi, replay : UniformReplay,
            ac_context : ACContext) -> None:
        self._loop(True, 0, env, replay, ac_context)


    def _train_interactions(self, interactions : int, env : Envi,
            replay : UniformReplay, ac_context : ACContext) -> None:
        self._loop(False, interactions, env, replay, ac_context)


    def _loop(self, explore : bool, interactions : int, env : Envi, replay : UniformReplay, ac_context : ACContext) -> None:
        model = ac_context.get_model()
        model_target = ac_context.get_model_target()
        model_opti = ac_context.get_model_opti()

        env.reset_steps()

        episode_iter = 0

        n_step_states_t = t.zeros(self.n_steps, self.frame_skip + 1,
            *env.obs_shape(), dtype = t.float)
        n_step_actions = t.zeros(self.n_steps, self.a_size, dtype = t.float)
        n_step_rewards = t.zeros(self.n_steps, 1, dtype = t.float)
        n_step_terminals = t.zeros(self.n_steps, 1, dtype = t.bool)

        while True:
            state_tt, _ = env.reset()

            terminal, trunc = False, False

            episode_iter += self.max_steps
            while env.steps_taken() < episode_iter:

                if explore:
                    if env.steps_taken() >= self.explore_steps:
                        return
                else:
                    if env.steps_taken() >= interactions:
                        return

                self._eval_mode(model)

                n_steps_taken = 0
                while n_steps_taken < self.n_steps:
                    
                    state_t = state_tt

                    if explore:
                        action = env.random_action()
                    else:
                        action = model(state_t.unsqueeze(0)).detach()
                        action = t.clip(action + self.noise.gen(1),
                            env.min_action(), env.max_action())

                    state_t = t.roll(state_t, shifts = -self.frame_skip,
                        dims = 0)

                    for skip in range(self.frame_skip):
                        state, _, terminal, trunc, _ = env.step(action)
                        state_t[skip + 1] = state

                        if terminal or trunc:
                            break

                    if terminal or trunc:
                        break

                    state, reward, terminal, trunc, _ = env.step(action)
                    state_tt = t.roll(state_t, shifts = -1, dims = 0)
                    state_tt[self.frame_skip] = state

                    if terminal or trunc:
                        break

                    n_step_states_t[n_steps_taken] = state_t
                    n_step_actions[n_steps_taken] = action
                    n_step_rewards[n_steps_taken] = reward
                    n_step_terminals[n_steps_taken] = terminal or trunc
                    n_steps_taken += 1


                reward_acc = 0.0

                view_states_t = n_step_states_t[:n_steps_taken]
                view_actions = n_step_actions[:n_steps_taken]
                view_rewards = n_step_rewards[:n_steps_taken]
                view_terminals = n_step_terminals[:n_steps_taken]

                for gamma_pwr, i in enumerate(range(n_steps_taken - 1, -1, -1)):
                    state_t = view_states_t[i]
                    action = view_actions[i]
                    prize = view_rewards[i]
                    term = view_terminals[i]

                    reward_acc = prize.item() + 0.99 * reward_acc

                    replay.store(state_t, action, reward_acc, state_tt,
                        bool(term), gamma_pwr + 1)

                if not explore:
                    current_steps = env.steps_taken()

                    for _ in range(n_steps_taken):
                        self._exploit_xp(env, replay, ac_context)

                    if self.perf.time_to_measure(current_steps):
                        self.perf.measure(current_steps, env, model)

                    if self.chk.time_to_save(current_steps):
                        self.chk.save(current_steps, model = model,
                                      model_target = model_target,
                                      model_opti = model_opti)


    def train(self, interactions : int, env : Envi, model : t.nn.Module,
            checkpoint : Optional[str] = None) -> None:

        replay = UniformReplay(frame_skip = self.frame_skip,
            state_size = env.obs_shape(), action_size = env.action_elems(),
            buffer_size = self.replay_size)

        self._train_mode(model)

        model_target = self._freeze_model(model)

        model_opti = t.optim.Adam(model.parameters(), lr = self.lr)

        if checkpoint:
            models = CheckpointManager.load(checkpoint,
                    model = model,
                    model_target = model_target,
                    model_opti = model_opti)

            assert isinstance(models["model"], t.nn.Module)
            assert isinstance(models["model_target"], t.nn.Module)
            assert isinstance(models["model_opti"], t.optim.Adam)

            model, model_target, model_opti = models["model"],\
                models["model_target"], models["model_opti"]

        ac_context = ACContext(model = model, model_target = model_target, model_opti = model_opti)

        self._explore(env, replay, ac_context)

        self._train_interactions(interactions, env, replay, ac_context)

        self.chk.save(interactions, model = model,
                        model_target = model_target,
                        model_opti = model_opti)

        self._eval_mode(model)
