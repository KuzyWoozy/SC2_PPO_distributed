import numpy as np
import torch as t

from pysc2.agents import base_agent
from pysc2.lib import actions

from src.Misc import categorical_sample
from src.Approximator import FDZApprox
from src.Config import LEARNING_RATE, PPO_CLIP, ENTROPY, LAMBDA


class RandomAgent(base_agent.BaseAgent):
    
    def __init__(self) -> None:
        super().__init__()

        self.acts = []

    def step(self, obs):
        super(RandomAgent, self).step(obs)
        function_id = np.random.choice(obs.observation.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
        
        if not function_id in self.acts:
            self.acts.append(function_id)

        print(args)

        return actions.FunctionCall(function_id, args)


class FDZAgent(base_agent.BaseAgent):

    def __init__(self, approx, check_manager = None) -> None:
        super().__init__()

        self.approx = approx

        self.optim = t.optim.Adam(self.approx.parameters(), maximize = False, lr = LEARNING_RATE)

        # Function.ability(453, "Stop_quick", cmd_quick, 3665)
        # Function.ability(334, "Patrol_minimap", cmd_minimap, 3795)
        # Function.ability(331, "Move_screen", cmd_screen, 3794)
        # Function.ability(451, "Smart_screen", cmd_screen, 1)
        # Function.ability(332, "Move_minimap", cmd_minimap, 3794)
        # Function.ability(12, "Attack_screen", cmd_screen, 3674)
        # Function.ui_func(3, "select_rect", select_rect)
        # Function.ui_func(4, "select_control_group", control_group)
        # Function.ui_func(1, "move_camera", move_camera)
        # Function.ability(333, "Patrol_screen", cmd_screen, 3795)
        # Function.ui_func(0, "no_op", no_op)
        # Function.ui_func(2, "select_point", select_point)
        # Function.ability(274, "HoldPosition_quick", cmd_quick, 3793)
        # Function.ability(13, "Attack_minimap", cmd_minimap, 3674)
        # Function.ui_func(7, "select_army", select_army,
        #                    lambda obs: obs.player_common.army_count > 0)
        # Function.ability(452, "Smart_minimap", cmd_minimap, 1)

        self.nn_starcraft_mapping = {
                         0 : 334,
                         1 : 451,
                         2 : 12,
                         3 : 3,
                         4 : 4,
                         5 : 333,
                         6 : 0,
                         7 : 2,
                         8 : 274,
                         9 : 13,
                         10 : 7,
                         11 : 452}

                         
        self.reverse_nn_starcraft_mapping = {v : k for k, v in self.nn_starcraft_mapping.items()}

        self.check_manager = check_manager
        

    def loss(self, func_args_dists, func_args_dists_old, actions, adv):
        # Note that we're minimizing
    
        actor_gain = t.tensor([0.0])
        entropy = t.tensor([0.0])
        
        adv_detached = adv.detach()

        for out, out_old, action in zip(func_args_dists, func_args_dists_old, actions):

            actor_gain -= t.min((out[:, action] / out_old[:, action]) * adv_detached, t.clip(out[:, action] / out_old[:, action], min = 1 - PPO_CLIP, max = 1 + PPO_CLIP) * adv_detached)

            out_pos = out[out > 0.0]
            entropy -= t.sum(out_pos * t.log(out_pos), axis = 0)

        critic_loss = adv ** 2
        
        return actor_gain + critic_loss + (ENTROPY * entropy)
    

    def mc_loss(self, episode_info):
        
        loss = t.tensor([0.0])
        G = 0.0

        for (reward, obs, func_args_dists_old, func_args_actions) in reversed(episode_info):
            func_args_dists, critic_val = self.nn_outs(obs, func_args_actions[0])
            G = reward + LAMBDA * G
            ADV = G - critic_val[0]
            
            loss += self.loss(func_args_dists, func_args_dists_old, func_args_actions, ADV)
        return loss



    def convert_to_state(self, obs):
        MAX_UNIT_HEURISTIC = 100

        state = t.from_numpy(np.expand_dims(np.stack((
                obs.observation.feature_screen.visibility_map / 3,
                obs.observation.feature_screen.player_id / 16,
                obs.observation.feature_screen.player_relative / 4,
                obs.observation.feature_screen.unit_type / MAX_UNIT_HEURISTIC,
                obs.observation.feature_screen.selected,
                obs.observation.feature_screen.unit_hit_points_ratio / 255,
                obs.observation.feature_screen.unit_density_aa / 255,

                obs.observation.feature_minimap.visibility_map / 3,
                obs.observation.feature_minimap.camera,
                obs.observation.feature_minimap.player_id / 16,
                obs.observation.feature_minimap.player_relative / 4,
                obs.observation.feature_minimap.selected), axis = 0), 0)).float()
        
        return state
       
        

    def step(self, obs):
        super().step(obs)
        
        state = self.convert_to_state(obs)
        
        nn_repr = self.approx(state, "hidden")

        (actor_probs,) = self.approx(nn_repr, "function_id")

        mask = t.zeros((1, self.approx.num_actions), dtype = t.float32)

        mask[:, t.tensor([self.reverse_nn_starcraft_mapping[act] for act in obs.observation.available_actions if act in self.reverse_nn_starcraft_mapping])] = 1.0
        
        actor_probs_masked = actor_probs * mask
        
        actor_probs_masked_norm = actor_probs_masked / t.sum(actor_probs_masked)

        actor_choice = categorical_sample(actor_probs_masked_norm)
        function_id = self.nn_starcraft_mapping[actor_choice]

        args, func_args_dists, func_args_actions = self.approx(nn_repr, function_id)
        func_args_dists.insert(0, actor_probs_masked_norm)
        func_args_actions.insert(0, actor_choice)

        return actions.FunctionCall(function_id, args), func_args_dists, func_args_actions


    def nn_outs(self, obs, actor_choice):
        state = self.convert_to_state(obs)

        nn_repr = self.approx(state, "hidden")

        (actor_probs,) = self.approx(nn_repr, "function_id")

        mask = t.zeros((1, self.approx.num_actions), dtype = t.float32)

        mask[:, t.tensor([self.reverse_nn_starcraft_mapping[act] for act in obs.observation.available_actions if act in self.reverse_nn_starcraft_mapping])] = 1.0
        
        actor_probs_masked = actor_probs * mask
        
        actor_probs_masked_norm = actor_probs_masked / t.sum(actor_probs_masked)
        function_id = self.nn_starcraft_mapping[actor_choice]
        
        _, func_args_dists, _ = self.approx(nn_repr, function_id)
        func_args_dists.insert(0, actor_probs_masked_norm)

        return func_args_dists, self.approx(nn_repr, "critic")
