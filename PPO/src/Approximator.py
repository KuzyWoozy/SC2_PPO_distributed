import torch as t
import copy

from src.Config import NN_HIDDEN_LAYER, LAMBDA


class FDZApprox(t.nn.Module):

    def __init__(self):
        super().__init__()

        self.convolve1 = t.nn.Conv2d(12, 16, 8, stride = 4)
        self.convolve2 = t.nn.Conv2d(16, 32, 4, stride = 2)

        self.actor_dense1 = t.nn.Linear(1152, NN_HIDDEN_LAYER)
        self.actor_dense2 = t.nn.Linear(NN_HIDDEN_LAYER, NN_HIDDEN_LAYER)

        self.function_id_policy = t.nn.Linear(NN_HIDDEN_LAYER, 11)

        self.argument_screen_x_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 64)
        self.argument_screen_y_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 64)        
        self.argument_minimap_x_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 64)
        self.argument_minimap_y_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 64)
        self.argument_screen2_x_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 64)
        self.argument_screen2_y_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 64)
        self.argument_select_point_act_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 4)
        self.argument_select_add_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 2)
        self.argument_select_unit_act_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 4)
        self.argument_select_unit_id_policy = t.nn.Linear(NN_HIDDEN_LAYER + 11, 500)
    
        self.critic = t.nn.Linear(NN_HIDDEN_LAYER, 1)

    
    def apply_conv_net(self, x):
        return t.flatten(t.relu(self.convolve2(t.relu(self.convolve1(x)))), start_dim = 1)


    def apply_hidden_net(self, x): 
        return t.relu(self.actor_dense2(t.relu(self.actor_dense1(x))))


    def forward(self, x, nn_type):
        
        if nn_type == "hidden":
            return self.apply_hidden_net(self.apply_conv_net(x))
        
        elif nn_type == "critic":
            return self.critic(x)

        elif nn_type == "function_id":
            return (t.softmax(self.function_id_policy(x), dim = 1),)
        
        elif nn_type == "screen":
            return t.softmax(self.argument_screen_x_policy(x), dim = 1), t.softmax(self.argument_screen_y_policy(x), dim = 1)

        elif nn_type == "minimap":
            return t.softmax(self.argument_minimap_x_policy(x), dim = 1), t.softmax(self.argument_minimap_y_policy(x), dim = 1)

        elif nn_type == "screen2":
            return t.softmax(self.argument_screen2_x_policy(x), dim = 1), t.softmax(self.argument_screen2_y_policy(x), dim = 1)

        elif nn_type == "select_point_act":
            return (t.softmax(self.argument_select_point_act_policy(x), dim = 1),)
        
        elif nn_type == "select_add":
            return (t.softmax(self.argument_select_add_policy(x), dim = 1),)
        
        elif nn_type == "select_unit_act":
            return (t.softmax(self.argument_select_unit_act_policy(x), dim = 1),)
        
        elif nn_type == "select_unit_id":
            return (t.softmax(self.argument_select_unit_id_policy(x), dim = 1),)
        
        else:
            print(f"{arg_type} IS NOT SUPPORTED")
            exit(1)
      

# Required because DDP is stupid, wrapper so top level forward-backward is performed once
class MonteCarloForwardWorkaround(t.nn.Module):

    def __init__(self, agent):
        super().__init__()

        self.agent = agent
        self.approx = agent.approx
    
    
    def forward(self, episode_info):
        
        G = 0.0
        loss = t.tensor([0.0])
        
        for (reward, obs, func_args_dists_old, func_args_actions) in reversed(episode_info):
            func_args_dists, critic_val = self.agent.nn_outs(obs, func_args_actions[0])
            
            G = reward + LAMBDA * G
            ADV = G - critic_val[0]
            
            loss += self.agent.loss(func_args_dists, func_args_dists_old, func_args_actions, ADV)
        
        return loss
