import numpy as np
import torch as t

from pysc2.agents import base_agent
from pysc2.lib import actions

from src.Misc import CheckpointManager, categorical_sample
from src.Config import MINIGAME_NAME, CHECK_INTERVAL, LEARNING_RATE, DTYPE, CHECK_LOAD

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



class MiniStarAgent(base_agent.BaseAgent):

    def __init__(self, policy) -> None:
        super().__init__()

        self.policy = policy

        self.optim = t.optim.Adam(policy.parameters(), maximize = False, lr = LEARNING_RATE)

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

        self.policy2function = {
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

                         
        self.function2policy = {v : k for k, v in self.policy2function.items()}

        self.check_manager = CheckpointManager("checkpoints", MINIGAME_NAME, CHECK_INTERVAL)
        
        if CHECK_LOAD:
            self.optim.load_state_dict(t.load(CHECK_LOAD)["optim"])
        

    def obs_to_state(self, obs):
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
                obs.observation.feature_minimap.selected), axis = 0), 0)).to(device = self.policy.get_device(), dtype = DTYPE)
        
        return state
       

    def step(self, obs):
        super().step(obs)
        
        state = self.obs_to_state(obs)
        
        nn_repr = self.policy(state, "hidden")

        (actor_probs,) = self.policy(nn_repr, "function_id")

        actor_probs = actor_probs.to(t.device("cpu"))

        mask = t.zeros((1, self.policy.get_num_actions()), dtype = DTYPE, device = t.device("cpu"))

        mask[:, t.tensor([self.function2policy[act] for act in obs.observation.available_actions if act in self.function2policy], device = t.device("cpu"))] = 1.0
        
        actor_probs_masked = actor_probs * mask
        
        actor_probs_masked_norm = actor_probs_masked / t.sum(actor_probs_masked)

        actor_choice = categorical_sample(actor_probs_masked_norm)
        function_id = self.policy2function[actor_choice]

        args, func_args_dists, func_args_actions = self.policy(nn_repr, function_id, sample = True)
        func_args_dists.insert(0, actor_probs_masked_norm.to(dtype = DTYPE, device = self.policy.get_device()))
        func_args_actions.insert(0, actor_choice)

        return actions.FunctionCall(function_id, args), func_args_dists, func_args_actions, self.policy(nn_repr, "critic")



    def nn_outs(self, obs, actor_choice):
        state = self.obs_to_state(obs)

        nn_repr = self.policy(state, "hidden")

        (actor_probs,) = self.policy(nn_repr, "function_id")

        mask = t.zeros((1, self.policy.get_num_actions()), dtype = DTYPE, device = self.policy.get_device())

        mask[:, t.tensor([self.function2policy[act] for act in obs.observation.available_actions if act in self.function2policy], device = self.policy.get_device())] = 1.0
        
        actor_probs_masked = actor_probs * mask
        
        actor_probs_masked_norm = actor_probs_masked / t.sum(actor_probs_masked)
        function_id = self.policy2function[actor_choice]
        
        func_args_dists = self.policy(nn_repr, function_id, sample = False)
        
        func_args_dists.insert(0, actor_probs_masked_norm)

        return func_args_dists, self.policy(nn_repr, "critic")

    def save_if_rdy(self, agent_steps):
        if self.check_manager.time_to_save(agent_steps):
            self.save(agent_steps)

    def save(self, agent_steps):
        self.check_manager.save(agent_steps, {"policy" : self.policy.get_state_dict(), "optim" : self.optim.state_dict()})

