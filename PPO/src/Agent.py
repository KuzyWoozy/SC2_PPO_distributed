import numpy as np
import torch as t

from pysc2.agents import base_agent
from pysc2.lib import actions

from src.Approximator import FDZApprox
from src.Config import LEARNING_RATE


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

        return actions.FunctionCall(function_id, args)


class FDZAgent(base_agent.BaseAgent):

    def __init__(self, check_manager = None) -> None:
        super().__init__()

        self.approx = FDZApprox()

        self.optim = t.optim.Adam(self.approx.parameters(), maximize = False, lr = LEARNING_RATE)

        # Function.ui_func(0, "no_op", no_op)
        # Function.ability(453, "Stop_quick", cmd_quick, 3665)
        # Function.ability(332, "Move_minimap", cmd_minimap, 3794) 
        # Function.ui_func(1, "move_camera", move_camera)
        # Function.ui_func(2, "select_point", select_point)
        # Function.ui_func(3, "select_rect", select_rect), 
        # Function.ability(13, "Attack_minimap", cmd_minimap, 3674)
        # Function.ability(331, "Move_screen", cmd_screen, 3794)
        # Function.ui_func(7, "select_army", select_army,
        #             lambda obs: obs.player_common.army_count > 0 
        # Function.ability(12, "Attack_screen", cmd_screen, 3674)
        # Function.ui_func(5, "select_unit", select_unit, lambda obs: obs.ui_data.HasField("multi")), 
        self.nn_starcraft_mapping = {0 : 0,
                         1 : 453, 
                         2 : 332,
                         3 : 1, 
                         4 : 2,
                         5 : 3,
                         6 : 13,
                         7 : 331,
                         8 : 7,
                         9 : 12,
                         10 : 5}

        self.reverse_nn_starcraft_mapping = {v : k for k, v in self.nn_starcraft_mapping.items()}

        self.check_manager = check_manager
        

    def categorical_sample(self, probs):
        return t.distributions.Categorical(probs = probs).sample((1,)).item()
    

    def convert_to_state(self, obs):
        MAX_UNIT_HEURISTIC = 105

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
       
        

    def step(self, obs, state):
        super().step(obs)

        nn_repr = self.approx.apply_hidden_net(self.approx.apply_conv_net(state))
        
        (actor_probs,) = self.approx("function_id", nn_repr)

        mask = t.zeros((1, 11), dtype = t.float32)

        mask[:, t.tensor([self.reverse_nn_starcraft_mapping[act] for act in obs.observation.available_actions if act in self.reverse_nn_starcraft_mapping])] = 1.0
        
        actor_probs_masked = actor_probs * mask
        
        actor_probs_masked_norm = actor_probs_masked / t.sum(actor_probs_masked)

        actor_choice = self.categorical_sample(actor_probs_masked_norm)
        function_id = self.nn_starcraft_mapping[actor_choice]

        actor_choice_vector = t.zeros((1, 11), dtype = t.float32)
        actor_choice_vector[:, actor_choice] = 1.0
        
        # Add selected function as an argument
        action_nn_repr = t.cat((actor_choice_vector, nn_repr), dim = 1)

        
        func_args_dists = [actor_probs_masked_norm]
        func_args_actions = [actor_choice]

        args = []

        for arg in self.action_spec.functions[function_id].args:
            if arg.name == "queued":
                args.append([0]) # Do not queue anything for now
            else:
                out = self.approx(arg.name, action_nn_repr)
                out_act = [self.categorical_sample(y) for y in out]

                func_args_dists.extend(out)
                func_args_actions.extend(out_act)

                args.append(out_act)


        return actions.FunctionCall(function_id, args), func_args_dists, func_args_actions, self.approx.critic(nn_repr)


    def step_old(self, approx_old, actor_choice, obs, state):

        nn_repr = approx_old.apply_hidden_net(approx_old.apply_conv_net(state))

        (actor_probs,) = approx_old("function_id", nn_repr)

        mask = t.zeros((1, 11), dtype = t.float32)

        mask[:, t.tensor([self.reverse_nn_starcraft_mapping[act] for act in obs.observation.available_actions if act in self.reverse_nn_starcraft_mapping])] = 1.0
        
        actor_probs_masked = actor_probs * mask
        
        actor_probs_masked_norm = actor_probs_masked / t.sum(actor_probs_masked)

        function_id = self.nn_starcraft_mapping[actor_choice]

        actor_choice_vector = t.zeros((1, 11), dtype = t.float32)
        actor_choice_vector[:, actor_choice] = 1.0
        
        # Add selected function as an argument
        action_nn_repr = t.cat((actor_choice_vector, nn_repr), dim = 1)

        
        func_args_dists = [actor_probs_masked_norm]

        args = []

        for arg in self.action_spec.functions[function_id].args:
            if arg.name == "queued":
                args.append([0]) # Do not queue anything for now
            else:
                func_args_dists.extend(approx_old(arg.name, action_nn_repr))

        return func_args_dists


    def critic(self, state):
        return self.approx.critic(self.approx.apply_hidden_net(self.approx.apply_conv_net(state)))

