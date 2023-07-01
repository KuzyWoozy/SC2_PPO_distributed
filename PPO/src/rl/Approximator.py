import torch as t

from src.Config import NN_HIDDEN_LAYER, NUM_ACTIONS, DTYPE
from src.Misc import categorical_sample

class AtariNet(t.nn.Module):

    def __init__(self):
        super().__init__()

        self.convolve1 = t.nn.Conv2d(5, 16, 8, stride = 4)
        self.convolve2 = t.nn.Conv2d(16, 32, 4, stride = 2)

        self.actor_dense1 = t.nn.Linear(1152, NN_HIDDEN_LAYER)

        self.function_id = t.nn.Linear(NN_HIDDEN_LAYER, NUM_ACTIONS)
        self.x1 = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.y1 = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.x2 = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.y2 = t.nn.Linear(NN_HIDDEN_LAYER, 64)
        self.select_control_group_act = t.nn.Linear(NN_HIDDEN_LAYER, 5)
        self.select_control_group_id = t.nn.Linear(NN_HIDDEN_LAYER, 10)
        self.select_point_add = t.nn.Linear(NN_HIDDEN_LAYER, 2)
        self.select_army_add = t.nn.Linear(NN_HIDDEN_LAYER, 2)
        self.critic = t.nn.Linear(NN_HIDDEN_LAYER, 1)

    
    def forward(self, inp):

        hid = t.relu(self.actor_dense1(t.flatten(t.relu(self.convolve2(t.relu(self.convolve1(inp)))), start_dim = 1)))

        return t.nn.functional.log_softmax(self.function_id(hid), dim = 1),\
                t.nn.functional.log_softmax(self.x1(hid), dim = 1),\
                t.nn.functional.log_softmax(self.y1(hid), dim = 1),\
                t.nn.functional.log_softmax(self.x2(hid), dim = 1),\
                t.nn.functional.log_softmax(self.y2(hid), dim = 1),\
                t.nn.functional.log_softmax(self.select_control_group_act(hid), dim = 1),\
                t.nn.functional.log_softmax(self.select_control_group_id(hid), dim = 1),\
                t.nn.functional.log_softmax(self.select_point_add(hid), dim = 1),\
                t.nn.functional.log_softmax(self.select_army_add(hid), dim = 1),\
                self.critic(hid)

    def sample_args(self, func_id, x1_prob, y1_prob, x2_prob, y2_prob, cg_act_prob, cg_id_prob, point_add_prob, army_add_prob, x1_prob_old, y1_prob_old, x2_prob_old, y2_prob_old, cg_act_prob_old, cg_id_prob_old, point_add_prob_old, army_add_prob_old):

        # Smart_screen
        if func_id == 451:
            x1_prob_cpu = x1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            y1_prob_cpu = y1_prob.to(dtype = DTYPE, device = t.device("cpu"))

            x1_choice = categorical_sample(x1_prob_cpu)
            y1_choice = categorical_sample(y1_prob_cpu)

            return [[0], [x1_choice, y1_choice]], [x1_prob, y1_prob], [x1_prob_old, y1_prob_old], [x1_choice, y1_choice]

        # Select_rect
        elif func_id == 3:
            x1_prob_cpu = x1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            y1_prob_cpu = y1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            x2_prob_cpu = x2_prob.to(dtype = DTYPE, device = t.device("cpu"))
            y2_prob_cpu = y2_prob.to(dtype = DTYPE, device = t.device("cpu"))


            x1_choice = categorical_sample(x1_prob_cpu)
            y1_choice = categorical_sample(y1_prob_cpu)
            x2_choice = categorical_sample(x2_prob_cpu)
            y2_choice = categorical_sample(y2_prob_cpu)

            return [[0], [x1_choice, y1_choice], [x2_choice, y2_choice]], [x1_prob, y1_prob, x2_prob, y2_prob], [x1_prob_old, y1_prob_old, x2_prob_old, y2_prob_old], [x1_choice, y1_choice, x2_choice, y2_choice]


        # Select_control_group
        elif func_id == 4:
            cg_act_prob_cpu = cg_act_prob.to(dtype = DTYPE, device = t.device("cpu"))
            cg_id_prob_cpu = cg_id_prob.to(dtype = DTYPE, device = t.device("cpu"))

            cg_act_choice = categorical_sample(cg_act_prob_cpu) 
            cg_id_choice = categorical_sample(cg_id_prob_cpu)
            return [[cg_act_choice], [cg_id_choice]], [cg_act_prob, cg_id_prob], [cg_act_prob_old, cg_id_prob_old], [cg_act_choice, cg_id_choice]

        
        # No_op
        elif func_id == 0:
            return [], [], [], []

        # Select_point
        elif func_id == 2:
            point_add_prob_cpu = point_add_prob.to(dtype = DTYPE, device = t.device("cpu"))
            x1_prob_cpu = x1_prob.to(dtype = DTYPE, device = t.device("cpu"))
            y1_prob_cpu = y1_prob.to(dtype = DTYPE, device = t.device("cpu"))

            point_add_choice = categorical_sample(point_add_prob_cpu)
            x1_choice = categorical_sample(x1_prob_cpu)
            y1_choice = categorical_sample(y1_prob_cpu)
            return [[point_add_choice], [x1_choice, y1_choice]], [point_add_prob, x1_prob, y1_prob], [point_add_prob_old, x1_prob_old, y1_prob_old], [point_add_choice, x1_choice, y1_choice]
            

        # HoldPosition_quick
        elif func_id == 274:
            return [[0]], [], [], []

        # Select army
        elif func_id == 7:
            army_add_prob_cpu = army_add_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            army_add_choice = categorical_sample(army_add_prob_cpu)
            return [[army_add_choice]], [army_add_prob], [army_add_prob_old], [army_add_choice]

        else:
            print(f"{func_id} IS NOT SUPPORTED")
            sys.exit(1)


class FullyConv(t.nn.Module):

    def __init__(self):
        super().__init__()

        self.convolve1 = t.nn.Conv2d(5, 16, 5, stride = 1, padding = "same")
        self.convolve2 = t.nn.Conv2d(16, 32, 3, stride = 1, padding = "same")

        self.hidden = t.nn.Linear(131072, NN_HIDDEN_LAYER)

        self.function_id = t.nn.Linear(NN_HIDDEN_LAYER, NUM_ACTIONS)

        self.coords1 = t.nn.Conv2d(32, 1, 1, stride = 1)
        self.coords2 = t.nn.Conv2d(32, 1, 1, stride = 1)

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

    def sample_args(self, func_id, coords1_prob, coords2_prob, cg_act_prob, cg_id_prob, point_add_prob, army_add_prob):

        # Smart_screen
        if func_id == 451:
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

        # Select army
        elif func_id == 7:
            army_add_prob_cpu = army_add_prob.to(dtype = DTYPE, device = t.device("cpu"))
            
            army_add_choice = categorical_sample(army_add_prob_cpu)
            return [[army_add_choice]], [army_add_prob], [army_add_choice]

        else:
            print(f"{func_id} IS NOT SUPPORTED")
            sys.exit(1)


    def probs_args(self, func_id, coord1_prob, coord2_prob, cg_act_prob, cg_id_prob, point_add_prob, army_add_prob):

        # Smart_screen
        if func_id == 451:
            return [coord1_prob]

        # Select_rect
        elif func_id == 3:
            return [coord1_prob, coord2_prob]

        # Select_control_group
        elif func_id == 4:
            return [cg_act_prob, cg_id_prob]
        
        # No_op
        elif func_id == 0:
            return []

        # Select_point
        elif func_id == 2:
            return [point_add_prob, coord1_prob]
            
        # HoldPosition_quick
        elif func_id == 274:
            return []

        # Select army
        elif func_id == 7:
            return [army_add_prob]

        else:
            print(f"{func_id} IS NOT SUPPORTED")
            sys.exit(1)


