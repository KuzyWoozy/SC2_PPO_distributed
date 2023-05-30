import torch as t
import abc
from typing import Any


class Envi(abc.ABC):
    
    def __init__(self, env : Any, frame_skip : int) -> None:
        super().__init__()
        self.env = env
        self.frame_skip = frame_skip
        self._steps_taken = 0
    
    def reset(self) -> tuple[t.Tensor, dict[str, Any]]:
        initial_state, info = self.env.reset(options={"randomize": False})
        initial_state = self.preprocess(t.from_numpy(initial_state).float())
        
        state = t.empty(self.frame_skip + 1, *initial_state.shape)
        state[:] = initial_state

        return state, info


    def step(self, action : t.Tensor) -> tuple[t.Tensor, int, bool, bool, dict[str, Any]]:
        
        state_tt, reward, terminal, trunc, info = self.env.step(action.squeeze().numpy())
        state_tt = self.preprocess(t.from_numpy(state_tt).float())
        
        self._steps_taken += 1

        return state_tt, reward, terminal, trunc, info

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
import torch as t
from typing import Any, Union

from src.nets import ActorCriticTwin



import torch as t


import gym, copy
import torch as t
from typing import Any, Optional, Union, overload

from src.nets import ActorCriticTwin
from src.replay import UniformReplay, Batch
from src.envi import Envi
from src.misc import CheckpointManager, GaussianNoise


# Simple container for QoL
class ACContext:
    def __init__(self, model : t.nn.Module, model_target : t.nn.Module, model_opti : t.optim.Adam) -> None:
        self.model = model
        self.model_target = model_target
        self.model_opti = model_opti

    def get_model(self) -> t.nn.Module:
        return self.model

    def get_model_target(self) -> t.nn.Module:
        return self.model_target

    def get_model_opti(self) -> t.optim.Adam:
        return self.model_opti


import torch as t
from typing import Any, Optional, Union, overload




class T3D:
    def __init__(self,
            lr : float,
            actor_delayed_update : int, 
            a_size : int,
            replay_size : int, max_steps : int, explore_steps : int, 
            exploit_every : int, frame_skip : int,
            batch_size : int, tau : float,
            chk : CheckpointManager,
            noise : GaussianNoise) -> None:
        
        self.lr = lr 
        
        self.actor_delayed_update = actor_delayed_update
       
        self.a_size = a_size

        self.replay_size = replay_size
        self.max_steps = max_steps
        self.explore_steps = explore_steps
        self.exploit_every = exploit_every
        self.frame_skip = frame_skip

        self.tau = tau
        self.batch_size = batch_size

        self.chk = chk
        self.noise = noise

        self.update_counter = 0
        # Params taken from T3D paper
        self.target_noise = GaussianNoise([a_size], 0.2)
        self.target_noise_cutoff = 0.5

    
    def _train_mode(self, *args : t.nn.Module) -> None:
        for arg in args:
            arg.train()


    def _eval_mode(self, *args : t.nn.Module) -> None:
        for arg in args:
            arg.eval()
    
    
    def _freeze_model(self, model : t.nn.Module) -> t.nn.Module:
        return copy.deepcopy(model).requires_grad_(False)


    def _sync_target_model(self, mdl : t.nn.Module, mdl_tar : t.nn.Module, tau : float = 0.001) -> None:
        
        for mdl_par, mdl_tar_par in zip(mdl.parameters(), mdl_tar.parameters()):
            mdl_tar_par.copy_(tau * mdl_par.detach() + (1 - tau) * mdl_tar_par) 


    def _step(self, opti : t.optim.Adam, loss : t.Tensor) -> None:
        opti.zero_grad()
        loss.backward()
        opti.step()
    
    def _regularize_action(self, model : t.nn.Module, env : Envi, state : t.Tensor) -> t.Tensor:
        return t.clip(model(state) + t.clip(self.target_noise.gen(self.batch_size), -self.target_noise_cutoff, self.target_noise_cutoff), env.min_action(), env.max_action())


    def _q_target(self, env : Envi, ac_context : ACContext, batch : Batch, gamma : float = 0.99) -> t.Tensor:
        
        model_target = ac_context.get_model_target()
   
        rewards = batch.get_rewards()
        states_tt = batch.get_states_tt()
        terminals = batch.get_terminals()

        action = self._regularize_action(model_target, env, states_tt)
        
        _, c_t, twin_c_t = model_target(states_tt, action)

        return rewards + (t.ones_like(terminals) - terminals) * gamma *\
            t.minimum(c_t, twin_c_t)


    def _update_critics(self, model : t.nn.Module, model_opti : t.optim.Adam, q_targets : t.Tensor, batch : Batch) -> None: 
        states_t = batch.get_states_t()
        actions = batch.get_actions()

        loss = model.critics_loss(states_t, actions, q_targets)

        self._step(model_opti, loss)


    def _update_actor(self, model : t.nn.Module, model_opti : t.optim.Adam, batch : Batch) -> None:
        states_t = batch.get_states_t()

        loss = model.actor_loss(states_t)
        
        self._step(model_opti, loss)


    def _explore(self, env : Envi, replay : UniformReplay) -> None:
        while(True):
            state_tt, _ = env.reset()

            for _ in range(self.max_steps):

                if env.steps_taken() >= self.explore_steps:
                    return

                state_t = state_tt

                action = env.random_action()

                state_t = t.roll(state_t, shifts = -self.frame_skip, dims = 0)
                for skip in range(self.frame_skip):
                    state, reward, terminal, trunc, _ = env.step(action)
                    state_t[skip + 1] = state 
                
                state, reward, terminal, trunc, _ = env.step(action)
                state_tt = t.roll(state_t, shifts = -1, dims = 0)
                state_tt[self.frame_skip] = state

                replay.store(state_t, action, reward, state_tt, terminal or trunc)

                if terminal or trunc:
                    break


    def _exploit_xp(self, env : Envi, replay : UniformReplay, ac_context : ACContext) -> None:
        model = ac_context.get_model()
        model_target = ac_context.get_model_target()
        model_opti = ac_context.get_model_opti()


        self._train_mode(model)

        batch = replay.sample(self.batch_size)

        q_targets = self._q_target(env, ac_context, batch)

        self._update_critics(model, model_opti, q_targets, batch)
        
        self.update_counter += 1
        if self.update_counter % self.actor_delayed_update == 0:
            self._update_actor(model, model_opti, batch)
            
            self._sync_target_model(mdl = model, mdl_tar = model_target, tau = self.tau)


    def _train_interactions(self, interactions : int, env : Envi, replay : UniformReplay, ac_context : ACContext) -> None:
        
        model = ac_context.get_model()
        model_target = ac_context.get_model_target()
        model_opti = ac_context.get_model_opti()
        
        env.reset_steps()
        
        while(True):
            state_tt, _ = env.reset()
            
            for step in range(self.max_steps):

                if env.steps_taken() >= interactions:
                    return
                 
                self._eval_mode(model)

                state_t = state_tt
                
                action = model(state_t.unsqueeze(0)).detach()
                action = t.clip(action + self.noise.gen(1), env.min_action(), env.max_action())
                state_t = t.roll(state_t, shifts = -self.frame_skip, dims = 0)
                
                for skip in range(self.frame_skip):
                    state, reward, terminal, trunc, _ = env.step(action)
                    state_t[skip + 1] = state 

                state, reward, terminal, trunc, _ = env.step(action)
                state_tt = t.roll(state_t, shifts = -1, dims = 0)
                state_tt[self.frame_skip] = state
                
                replay.store(state_t, action, reward, state_tt, terminal or trunc)
               
                if (step + 1) % self.exploit_every == 0:
                    for _ in range(self.exploit_every):
                        self._exploit_xp(env, replay, ac_context)

                if self.chk.time_to_save(env.steps_taken()):
                    self.chk.save(env.steps_taken(), model = model,
                                  model_target = model_target,
                                  model_opti = model_opti) 

                if terminal or trunc:
                    break


    def train(self, interactions : int, env : Envi, model : ActorCriticTwin, checkpoint : Optional[str] = None) -> None:
        
        replay = UniformReplay(frame_skip = self.frame_skip, state_size = env.obs_shape(), \
            action_size = env.action_elems(), buffer_size = self.replay_size)

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
               
            critic, model_target, model_opti = models["model"], models["model_target"], models["model_opti"],

        ac_context = ACContext(model = model, model_target = model_target, model_opti = model_opti)
       
        self._explore(env, replay)
        
        self._train_interactions(interactions, env, replay, ac_context)

        self.chk.save(interactions, model = model,
                        model_target = model_target,
                        model_opti = model_opti)

        self._eval_mode(model)

