import torch as t
import copy

from src.Config import NN_HIDDEN_LAYER, PPO_CLIP, ENTROPY


class FDZApprox(t.nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size = NN_HIDDEN_LAYER

        self.convolve1 = t.nn.Conv2d(12, 16, 8, stride = 4)
        self.convolve2 = t.nn.Conv2d(16, 32, 4, stride = 2)

        self.actor_dense1 = t.nn.Linear(1152, self.hidden_size)
        self.actor_dense2 = t.nn.Linear(self.hidden_size, self.hidden_size)

        self.function_id_policy = t.nn.Linear(self.hidden_size, 11)

        self.argument_screen_x_policy = t.nn.Linear(self.hidden_size + 11, 64)
        self.argument_screen_y_policy = t.nn.Linear(self.hidden_size + 11, 64)        
        self.argument_minimap_x_policy = t.nn.Linear(self.hidden_size + 11, 64)
        self.argument_minimap_y_policy = t.nn.Linear(self.hidden_size + 11, 64)
        self.argument_screen2_x_policy = t.nn.Linear(self.hidden_size + 11, 64)
        self.argument_screen2_y_policy = t.nn.Linear(self.hidden_size + 11, 64)
        self.argument_select_point_act_policy = t.nn.Linear(self.hidden_size + 11, 4)
        self.argument_select_add_policy = t.nn.Linear(self.hidden_size + 11, 2)
        self.argument_select_unit_act_policy = t.nn.Linear(self.hidden_size + 11, 4)
        self.argument_select_unit_id_policy = t.nn.Linear(self.hidden_size + 11, 500)
    
        self.critic = t.nn.Linear(self.hidden_size, 1)


    def apply_conv_net(self, x):
        return t.flatten(t.relu(self.convolve2(t.relu(self.convolve1(x)))), start_dim = 1)


    def apply_hidden_net(self, x): 
        return t.relu(self.actor_dense2(t.relu(self.actor_dense1(x))))


    def forward(self, arg_type, x):
        
        if arg_type == "function_id":
            return (t.softmax(self.function_id_policy(x), dim = 1),)
        
        elif arg_type == "screen":
            return t.softmax(self.argument_screen_x_policy(x), dim = 1), t.softmax(self.argument_screen_y_policy(x), dim = 1)

        elif arg_type == "minimap":
            return t.softmax(self.argument_minimap_x_policy(x), dim = 1), t.softmax(self.argument_minimap_y_policy(x), dim = 1)

        elif arg_type == "screen2":
            return t.softmax(self.argument_screen2_x_policy(x), dim = 1), t.softmax(self.argument_screen2_y_policy(x), dim = 1)

        elif arg_type == "select_point_act":
            return (t.softmax(self.argument_select_point_act_policy(x), dim = 1),)
        
        elif arg_type == "select_add":
            return (t.softmax(self.argument_select_add_policy(x), dim = 1),)
        
        elif arg_type == "select_unit_act":
            return (t.softmax(self.argument_select_unit_act_policy(x), dim = 1),)
        
        elif arg_type == "select_unit_id":
            return (t.softmax(self.argument_select_unit_id_policy(x), dim = 1),)
        
        else:
            print(f"{arg.name} IS NOT SUPPORTED")
            exit(1)


    def loss(self, func_args_dists, actions, func_args_dists_old, adv):
        # Note that we're minimizing

        actor_gain = t.tensor([0.0])
        entropy = t.tensor([0.0])

        adv_detached = adv.detach()

        for out, action, out_old in zip(func_args_dists, actions, func_args_dists_old):

            actor_gain -= t.min((out[:, action] / out_old[:, action]) * adv_detached, t.clip(out[:, action] / out_old[:, action], min = 1 - PPO_CLIP, max = 1 + PPO_CLIP) * adv_detached)

            out_pos = out[out > 0.0]
            entropy -= t.sum(out_pos * t.log(out_pos), axis = 0)

        critic_loss = adv ** 2
        
        return actor_gain + critic_loss + (ENTROPY * entropy)
