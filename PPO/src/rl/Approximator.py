import torch as t

from src.Config import NN_HIDDEN_LAYER, NUM_ACTIONS


class MiniStarPolicy(t.nn.Module):

    def __init__(self):
        super().__init__()

        self.convolve1 = t.nn.Conv2d(12, 6, 5, stride = 1, padding = "same")
        self.convolve2 = t.nn.Conv2d(6, 8, 3, stride = 1, padding = "same")

        self.hidden = t.nn.Linear(32768, NN_HIDDEN_LAYER)
        
        self.function_id = t.nn.Linear(NN_HIDDEN_LAYER, NUM_ACTIONS)
        
        self.coords1 = t.nn.Conv2d(8, 1, 1, stride = 1)
        self.coords2 = t.nn.Conv2d(8, 1, 1, stride = 1)

        self.select_control_group_act = t.nn.Linear(NN_HIDDEN_LAYER, 5)
        self.select_control_group_id = t.nn.Linear(NN_HIDDEN_LAYER, 10)
        self.select_point_add = t.nn.Linear(NN_HIDDEN_LAYER, 2)
        self.select_army_add = t.nn.Linear(NN_HIDDEN_LAYER, 2)
        self.critic = t.nn.Linear(NN_HIDDEN_LAYER, 1)

    
    def forward(self, inp):
        
        convolution = t.relu(self.convolve2(t.relu(self.convolve1(inp))))

        hid = t.relu(self.hidden(t.flatten(convolution, start_dim = 1)))

        return t.softmax(self.function_id(hid), dim = 1),\
                t.softmax(t.flatten(self.coords1(convolution), start_dim = 1), dim = 1),\
                t.softmax(t.flatten(self.coords2(convolution), start_dim = 1), dim = 1),\
                t.softmax(self.select_control_group_act(hid), dim = 1),\
                t.softmax(self.select_control_group_id(hid), dim = 1),\
                t.softmax(self.select_point_add(hid), dim = 1),\
                t.softmax(self.select_army_add(hid), dim = 1),\
                self.critic(hid)
