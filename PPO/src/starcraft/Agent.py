import sys
import numpy as np
import torch as t

from pysc2.agents import base_agent
from pysc2.lib import actions

from src.Misc import CheckpointManager, categorical_sample
from src.Config import MINIGAME_NAME, CHECK_INTERVAL, LEARNING_RATE, DTYPE, CHECK_LOAD, GPU, NUM_ACTIONS


class MiniStarAgent(base_agent.BaseAgent):

    def __init__(self, policy) -> None:
        super().__init__()

        self.policy = policy

        self.optim = t.optim.Adam(policy.parameters(), maximize = False, lr = LEARNING_RATE)
        
        # Function.ability(451, "Smart_screen", cmd_screen, 1)
        # Function.ui_func(3, "select_rect", select_rect)
        # Function.ui_func(4, "select_control_group", control_group)
        # Function.ui_func(0, "no_op", no_op)
        # Function.ui_func(2, "select_point", select_point)
        # Function.ability(274, "HoldPosition_quick", cmd_quick, 3793)
        # Function.ui_func(7, "select_army", select_army,
        
        self.policy2function = {
                         0 : 451,
                         1 : 3,
                         2 : 4,
                         3 : 0,
                         4 : 2,
                         5 : 274,
                         6 : 7}

                         
        self.function2policy = {v : k for k, v in self.policy2function.items()}

        self.check_manager = CheckpointManager("checkpoints", MINIGAME_NAME, CHECK_INTERVAL)
        
        if CHECK_LOAD:
            self.optim.load_state_dict(t.load(CHECK_LOAD)["optim"])
        
    def obs_to_state(self, obs):
        MAX_UNIT_HEURISTIC = 100

        state = t.from_numpy(np.expand_dims(np.stack((
                #obs.observation.feature_screen.visibility_map / 3,
                obs.observation.feature_screen.player_relative / 4,
                obs.observation.feature_screen.unit_type,
                obs.observation.feature_screen.selected,
                obs.observation.feature_screen.unit_hit_points_ratio / 255,
                obs.observation.feature_screen.unit_density_aa / 255

                #obs.observation.feature_minimap.visibility_map / 3,
                #obs.observation.feature_minimap.camera,
                #obs.observation.feature_minimap.player_id / 16,
                #obs.observation.feature_minimap.player_relative / 4,
                #obs.observation.feature_minimap.selected

                ), axis = 0), 0)).type(DTYPE)

        if GPU:
            state = state.to(device = self.policy.device)

        return state
       
    
    def step(self, obs):
        super().step(obs)
        
        state = self.obs_to_state(obs)
        

        mask = t.zeros((1, NUM_ACTIONS), dtype = DTYPE, device = t.device("cpu"))
        mask[:, [self.function2policy[act] for act in obs.observation.available_actions if act in self.function2policy]] = 1.0

        mask = mask.to(dtype = DTYPE, device = self.policy.device)

        policy_distributions = self.policy(state)

        actor_prob_masked = policy_distributions[0] * mask
        actor_prob_masked_norm = (actor_prob_masked / t.sum(actor_prob_masked))
        actor_prob_masked_norm_cpu = actor_prob_masked_norm.to(dtype = DTYPE, device = t.device("cpu"))
        actor_choice = categorical_sample(actor_prob_masked_norm_cpu)
        function_id = self.policy2function[actor_choice]
    
        
        print(actor_prob_masked_norm_cpu)

        args, args_probs, args_flat = self.policy.sample_args(function_id, *policy_distributions[1:-1])

        args_probs.insert(0, actor_prob_masked_norm)
        args_flat.insert(0, actor_choice)
        
        return actions.FunctionCall(function_id, args), args_probs, args_flat, mask, policy_distributions[-1]

    
    def nn_outs(self, state, mask, actor_choice):
        policy_distributions = self.policy(state)

        actor_prob_masked = policy_distributions[0] * mask        
        actor_prob_masked_norm = actor_prob_masked / t.sum(actor_prob_masked)
        function_id = self.policy2function[actor_choice]
        
        args_probs = self.policy.probs_args(function_id, *policy_distributions[1:-1])
        
        args_probs.insert(0, actor_prob_masked_norm)

        return args_probs, policy_distributions[-1]


    def save_if_rdy(self, agent_steps):
        if self.check_manager.time_to_save(agent_steps):
            self.save(agent_steps)

    def save(self, agent_steps):
        self.check_manager.save(agent_steps, {"policy" : self.policy.get_state_dict(), "optim" : self.optim.state_dict()})



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

        print("FOUND ARGS:", self.acts)

        return actions.FunctionCall(function_id, args), [], [], [], []

