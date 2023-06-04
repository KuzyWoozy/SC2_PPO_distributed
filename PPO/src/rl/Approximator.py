import torch as t
import copy

from src.Misc import categorical_sample
from src.Config import NN_HIDDEN_LAYER


class MiniStarPolicy(t.nn.Module):

    def __init__(self):
        super().__init__()

        self.num_actions = 12

        self.convolve1 = t.nn.Conv2d(12, 16, 8, stride = 4)
        self.convolve2 = t.nn.Conv2d(16, 32, 4, stride = 2)

        self.actor_dense1 = t.nn.Linear(1152, NN_HIDDEN_LAYER)
        self.actor_dense2 = t.nn.Linear(NN_HIDDEN_LAYER, NN_HIDDEN_LAYER)

        self.function_id_policy = t.nn.Linear(NN_HIDDEN_LAYER, self.num_actions)
        self.patrol_minimap_x_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.patrol_minimap_y_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        
        self.smart_screen_x_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.smart_screen_y_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        
        self.attack_screen_x_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.attack_screen_y_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        
        self.select_rect_x1_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.select_rect_y1_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.select_rect_x2_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.select_rect_y2_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)

        self.select_control_group_act_policy = t.nn.Linear(NN_HIDDEN_LAYER, 5)
        self.select_control_group_id_policy = t.nn.Linear(NN_HIDDEN_LAYER, 10)

        self.patrol_screen_x_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64) 
        self.patrol_screen_y_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)

        self.select_point_add_policy = t.nn.Linear(NN_HIDDEN_LAYER, 2)
        self.select_point_x_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.select_point_y_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)

        self.attack_minimap_x_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.attack_minimap_y_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)

        self.select_army_add_policy = t.nn.Linear(NN_HIDDEN_LAYER, 2)

        self.smart_minimap_x_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.smart_minimap_y_policy = t.nn.Linear(NN_HIDDEN_LAYER, 64)

        self.critic = t.nn.Linear(NN_HIDDEN_LAYER, 1)



    
    def apply_conv_net(self, inp):
        return t.flatten(t.relu(self.convolve2(t.relu(self.convolve1(inp)))), start_dim = 1)

    def apply_hidden_net(self, inp): 
        return t.relu(self.actor_dense2(t.relu(self.actor_dense1(inp))))
    
    def get_num_actions(self):
        return self.num_actions
    
    def forward(self, inp, nn_type, sample = False):

        if nn_type == "hidden":
            return self.apply_hidden_net(self.apply_conv_net(inp))
        
        elif nn_type == "critic":
            return self.critic(inp)

        elif nn_type == "function_id":
            return (t.softmax(self.function_id_policy(inp), dim = 1),)

        # Patrol_minimap
        elif nn_type == 334:
            x_dist = t.softmax(self.patrol_minimap_x_policy(inp), dim = 1)
            y_dist = t.softmax(self.patrol_minimap_y_policy(inp), dim = 1)
            
            if sample:
                x_choice = categorical_sample(x_dist)
                y_choice = categorical_sample(y_dist)
            
                return [[0], [x_choice, y_choice]], [x_dist, y_dist], [x_choice, y_choice]
            return [x_dist, y_dist]

        # Smart_screen
        elif nn_type == 451:
            x_dist = t.softmax(self.smart_screen_x_policy(inp), dim = 1)
            y_dist = t.softmax(self.smart_screen_y_policy(inp), dim = 1)

            if sample:
                x_choice = categorical_sample(x_dist)
                y_choice = categorical_sample(y_dist) 

                return [[0], [x_choice, y_choice]], [x_dist, y_dist], [x_choice, y_choice]
            return [x_dist, y_dist]

        # Attack_screen
        elif nn_type == 12:
            x_dist = t.softmax(self.attack_screen_x_policy(inp), dim = 1)
            y_dist = t.softmax(self.attack_screen_y_policy(inp), dim = 1)
            
            if sample:
                x_choice = categorical_sample(x_dist)
                y_choice = categorical_sample(y_dist)

                return [[0], [x_choice, y_choice]], [x_dist, y_dist], [x_choice, y_choice]
            return [x_dist, y_dist]

        # Select_rect
        elif nn_type == 3:
            x1_dist = t.softmax(self.select_rect_x1_policy(inp), dim = 1)
            y1_dist = t.softmax(self.select_rect_y1_policy(inp), dim = 1)
            x2_dist = t.softmax(self.select_rect_x2_policy(inp), dim = 1)
            y2_dist = t.softmax(self.select_rect_y2_policy(inp), dim = 1)

            if sample:
                x1_choice = categorical_sample(x1_dist)
                y1_choice = categorical_sample(y1_dist) 
                x2_choice = categorical_sample(x2_dist)
                y2_choice = categorical_sample(y2_dist)

                return [[0], [x1_choice, y1_choice], [x2_choice, y2_choice]], [x1_dist, y1_dist, x2_dist, y2_dist], [x1_choice, y1_choice, x2_choice, y2_choice]
            
            return [x1_dist, y1_dist, x2_dist, y2_dist]


        # Select_control_group
        elif nn_type == 4:
            group_act_dist = t.softmax(self.select_control_group_act_policy(inp), dim = 1)
            group_id_dist = t.softmax(self.select_control_group_id_policy(inp), dim = 1)
            if sample:
                group_act_choice = categorical_sample(group_act_dist) 
                group_id_choice = categorical_sample(group_id_dist)

                return [[group_act_choice], [group_id_choice]], [group_act_dist, group_id_dist], [group_act_choice, group_id_choice]

            return [group_act_dist, group_id_dist]
        
        # Patrol_screen 
        elif nn_type == 333:
            x_dist = t.softmax(self.patrol_screen_x_policy(inp), dim = 1)
            y_dist = t.softmax(self.patrol_screen_y_policy(inp), dim = 1)
            
            if sample:
                x_choice = categorical_sample(x_dist)
                y_choice = categorical_sample(y_dist)
                
                return [[0], [x_choice, y_choice]], [x_dist, y_dist], [x_choice, y_choice]
            return [x_dist, y_dist]

        # No_op
        elif nn_type == 0:
            if sample:
                return [], [], []
            return []

        # Select_point
        elif nn_type == 2:
            select_add_dist = t.softmax(self.select_point_add_policy(inp), dim = 1)
            x_dist = t.softmax(self.select_point_x_policy(inp), dim = 1)
            y_dist = t.softmax(self.select_point_y_policy(inp), dim = 1)
            
            if sample:
                select_add_choice = categorical_sample(select_add_dist)
                x_choice = categorical_sample(x_dist)
                y_choice = categorical_sample(y_dist)

                return [[select_add_choice], [x_choice, y_choice]], [select_add_dist, x_dist, y_dist], [select_add_choice, x_choice, y_choice]
            
            return [select_add_dist, x_dist, y_dist]


        # HoldPosition_quick
        elif nn_type == 274:
            if sample:
                return [[0]], [], []
            return []

        # Attack_minimap
        elif nn_type == 13:
            x_dist = t.softmax(self.attack_minimap_x_policy(inp), dim = 1)
            y_dist = t.softmax(self.attack_minimap_y_policy(inp), dim = 1)
            
            if sample:
                x_choice = categorical_sample(x_dist)
                y_choice = categorical_sample(y_dist)

                return [[0], [x_choice, y_choice]], [x_dist, y_dist], [x_choice, y_choice]
            return [x_dist, y_dist]

        # Select army
        elif nn_type == 7:
            select_add_dist = t.softmax(self.select_army_add_policy(inp), dim = 1)

            if sample:
                select_add_choice = categorical_sample(select_add_dist)

                return [[select_add_choice]], [select_add_dist], [select_add_choice]
            
            return [select_add_dist]

        # Smart minimap
        elif nn_type == 452:
            x_dist = t.softmax(self.smart_minimap_x_policy(inp), dim = 1)
            y_dist = t.softmax(self.smart_minimap_y_policy(inp), dim = 1)

            if sample:
                x_choice = categorical_sample(x_dist)
                y_choice = categorical_sample(y_dist) 

                return [[0], [x_choice, y_choice]], [x_dist, y_dist], [x_choice, y_choice]
            return [x_dist, y_dist]

        else:
            print(f"{nn_type} IS NOT SUPPORTED")
            exit(1)


    def get_state_dict(self):
        return self.state_dict()
