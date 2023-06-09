import torch as t
import copy

from src.Config import NN_HIDDEN_LAYER, DTYPE


class MiniStarPolicy(t.nn.Module):

    def __init__(self):
        super().__init__()

        self.num_actions = 12

        self.convolve1 = t.nn.Conv2d(12, 16, 8, stride = 4)
        self.convolve2 = t.nn.Conv2d(16, 32, 4, stride = 2)

        self.actor_dense1 = t.nn.Linear(1152, NN_HIDDEN_LAYER)
        self.actor_dense2 = t.nn.Linear(NN_HIDDEN_LAYER, NN_HIDDEN_LAYER)


        self.function_id = t.nn.Linear(NN_HIDDEN_LAYER, self.num_actions)

        self.x1 = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.y1 = t.nn.Linear(NN_HIDDEN_LAYER, 64)

        self.x2 = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.y2 = t.nn.Linear(NN_HIDDEN_LAYER, 64)

        self.select_control_group_act = t.nn.Linear(NN_HIDDEN_LAYER, 5)
        self.select_control_group_id = t.nn.Linear(NN_HIDDEN_LAYER, 10)

        self.select_point_add = t.nn.Linear(NN_HIDDEN_LAYER, 2)

        self.select_army_add_policy = t.nn.Linear(NN_HIDDEN_LAYER, 2)

        self.critic = t.nn.Linear(NN_HIDDEN_LAYER, 1)


    def get_num_actions(self):
        return self.num_actions


    def forward(self, inp):

        hid = t.relu(self.actor_dense2(t.relu(self.actor_dense1(t.flatten(t.relu(self.convolve2(t.relu(self.convolve1(inp)))), start_dim = 1)))))

        return t.softmax(self.function_id(hid), dim = 1),\
                t.softmax(self.x1(hid), dim = 1),\
                t.softmax(self.y1(hid), dim = 1),\
                t.softmax(self.x2(hid), dim = 1),\
                t.softmax(self.y2(hid), dim = 1),\
                t.softmax(self.select_control_group_act(hid), dim = 1),\
                t.softmax(self.select_control_group_id(hid), dim = 1),\
                t.softmax(self.select_point_add(hid), dim = 1),\
                t.softmax(self.select_army_add_policy(hid), dim = 1),\
                t.softmax(self.critic(hid), dim = 1)
        
    def get_state_dict(self):
        return self.state_dict()
