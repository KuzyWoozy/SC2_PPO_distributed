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
                obs.observation.feature_minimap.selected), axis = 0), 0)).type(DTYPE)
        
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

        args, args_probs, args_flat = self.sample_args(function_id, *policy_distributions[1:-1])

        args_probs.insert(0, actor_prob_masked_norm)
        args_flat.insert(0, actor_choice)
        
        return actions.FunctionCall(function_id, args), args_probs, args_flat, mask, policy_distributions[-1]

    
    def nn_outs(self, state, mask, actor_choice):
        policy_distributions = self.policy(state)

        actor_prob_masked = policy_distributions[0] * mask        
        actor_prob_masked_norm = actor_prob_masked / t.sum(actor_prob_masked)
        function_id = self.policy2function[actor_choice]
        
        args_probs = self.probs_args(function_id, *policy_distributions[1:-1])
        
        args_probs.insert(0, actor_prob_masked_norm)

        return args_probs, policy_distributions[-1]


    def save_if_rdy(self, agent_steps):
        if self.check_manager.time_to_save(agent_steps):
            self.save(agent_steps)


    def save(self, agent_steps):
        self.check_manager.save(agent_steps, {"policy" : self.policy.get_state_dict(), "optim" : self.optim.state_dict()})

    def sample_args(self, func_id, coords1_prob, coords2_prob, cg_act_prob, cg_id_prob, point_add_prob, army_add_prob):

        # Patrol_minimap
        if func_id == 334:
            coords1_prob_cpu = coords1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            coords1_choice = categorical_sample(coords1_prob_cpu)
            
            x1_choice = coords1_choice % 64
            y1_choice = coords1_choice // 64
            
            return [[0], [x1_choice, y1_choice]], [coords1_prob], [coords1_choice]

        # Smart_screen
        elif func_id == 451:
            coords1_prob_cpu = coords1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            coords1_choice = categorical_sample(coords1_prob_cpu)
            
            x1_choice = coords1_choice % 64
            y1_choice = coords1_choice // 64
            
            return [[0], [x1_choice, y1_choice]], [coords1_prob], [coords1_choice]

        # Attack_screen
        elif func_id == 12:
            coords1_prob_cpu = coords1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            coords1_choice = categorical_sample(coords1_prob_cpu)
            
            x1_choice = coords1_choice % 64
            y1_choice = coords1_choice // 64

            return [[0], [x1_choice, y1_choice]], [coords1_prob], [coords1_choice]

        # Select_rect
        elif func_id == 3:
            
            coords1_prob_cpu = coords1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            coords2_prob_cpu = coords2_prob.to(dtype = DTYPE, device = t.device("cpu"))

            coords1_choice = categorical_sample(coords1_prob_cpu)
            coords2_choice = categorical_sample(coords2_prob_cpu)

            x1_choice = coords1_choice % 64
            y1_choice = coords1_choice // 64
            x2_choice = coords2_choice % 64
            y2_choice = coords2_choice // 64

            
            return [[0], [x1_choice, y1_choice], [x2_choice, y2_choice]], [coords1_prob, coords2_prob], [coords1_choice, coords2_choice]


        # Select_control_group
        elif func_id == 4:
            cg_act_prob_cpu = cg_act_prob.to(dtype = DTYPE, device = t.device("cpu"))
            cg_id_prob_cpu = cg_id_prob.to(dtype = DTYPE, device = t.device("cpu"))

            cg_act_choice = categorical_sample(cg_act_prob_cpu) 
            cg_id_choice = categorical_sample(cg_id_prob_cpu)
            return [[cg_act_choice], [cg_id_choice]], [cg_act_prob, cg_id_prob], [cg_act_choice, cg_id_choice]

        
        # Patrol_screen 
        elif func_id == 333:
            coords1_prob_cpu = coords1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            coords1_choice = categorical_sample(coords1_prob_cpu)
            
            x1_choice = coords1_choice % 64
            y1_choice = coords1_choice // 64
            
            return [[0], [x1_choice, y1_choice]], [coords1_prob], [coords1_choice]


        # No_op
        elif func_id == 0:
            return [], [], []

        # Select_point
        elif func_id == 2:
            point_add_prob_cpu = point_add_prob.to(dtype = DTYPE, device = t.device("cpu"))
            coords1_prob_cpu = coords1_prob.to(dtype = DTYPE, device = t.device("cpu"))

            point_add_choice = categorical_sample(point_add_prob_cpu)
            coords1_choice = categorical_sample(coords1_prob_cpu)
            
            x1_choice = coords1_choice % 64
            y1_choice = coords1_choice // 64
            
            return [[point_add_choice], [x1_choice, y1_choice]], [point_add_prob, coords1_prob], [point_add_choice, coords1_choice]
            

        # HoldPosition_quick
        elif func_id == 274:
            return [[0]], [], []

        # Attack_minimap
        elif func_id == 13:
            coords1_prob_cpu = coords1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            coords1_choice = categorical_sample(coords1_prob_cpu)
            
            x1_choice = coords1_choice % 64
            y1_choice = coords1_choice // 64
            
            return [[0], [x1_choice, y1_choice]], [coords1_prob], [coords1_choice]

        # Select army
        elif func_id == 7:
            army_add_prob_cpu = army_add_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            army_add_choice = categorical_sample(army_add_prob_cpu)
            return [[army_add_choice]], [army_add_prob], [army_add_choice]

        # Smart minimap
        elif func_id == 452:
            coords1_prob_cpu = coords1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            coords1_choice = categorical_sample(coords1_prob_cpu)
            
            x1_choice = coords1_choice % 64
            y1_choice = coords1_choice // 64
            
            return [[0], [x1_choice, y1_choice]], [coords1_prob], [coords1_choice]

        else:
            print(f"{func_id} IS NOT SUPPORTED")
            sys.exit(1)


    def probs_args(self, func_id, coord1_prob, coord2_prob, cg_act_prob, cg_id_prob, point_add_prob, army_add_prob):

        # Patrol_minimap
        if func_id == 334:
            return [coord1_prob]

        # Smart_screen
        elif func_id == 451:
            return [coord1_prob]

        # Attack_screen
        elif func_id == 12:
            return [coord1_prob]

        # Select_rect
        elif func_id == 3:
            return [coord1_prob, coord2_prob]

        # Select_control_group
        elif func_id == 4:
            return [cg_act_prob, cg_id_prob]
        
        # Patrol_screen 
        elif func_id == 333:
            return [coord1_prob]

        # No_op
        elif func_id == 0:
            return []

        # Select_point
        elif func_id == 2:
            return [point_add_prob, coord1_prob]
            
        # HoldPosition_quick
        elif func_id == 274:
            return []

        # Attack_minimap
        elif func_id == 13:
            return [coord1_prob]

        # Select army
        elif func_id == 7:
            return [army_add_prob]

        # Smart minimap
        elif func_id == 452:
            return [coord1_prob]

        else:
            print(f"{func_id} IS NOT SUPPORTED")
            sys.exit(1)



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
