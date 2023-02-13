from typing import Any, Optional, Union, overload

import torch as t


class ActorCriticTwin(t.nn.Module):
    def __init__(self, frame_skip : int, h : int, o : int) -> None:
        super().__init__()

        self.hidden_size = h
        self.output_size = o

        self.conv1 = t.nn.Conv2d(frame_skip + 1, 8, 7, stride = 6, padding = 1)
        self.conv2 = t.nn.Conv2d(8, 12, 5, stride = 4, padding = 2)
        self.conv3 = t.nn.Conv2d(12, 16, 3, stride = 1)


        self.crit_dense1 = t.nn.Linear(64 + o, h)
        self.crit_dense2 = t.nn.Linear(h, h)
        self.critic = t.nn.Linear(h, 1)

        self.crit_twin_dense1 = t.nn.Linear(64 + o, h)
        self.crit_twin_dense2 = t.nn.Linear(h, h)
        self.critic_twin = t.nn.Linear(h, 1)

        self.act_dense1 = t.nn.Linear(64, h)
        self.act_dense2 = t.nn.Linear(h, h)
        self.actor = t.nn.Linear(h, o)


    @overload
    def forward(self, states : t.Tensor, actions : None) -> t.Tensor: ...


    @overload
    def forward(self, states : t.Tensor, actions : t.Tensor)\
            -> tuple[t.Tensor, t.Tensor, t.Tensor]: ...


    def forward(self, states : t.Tensor, actions : Optional[t.Tensor] = None)\
            -> Union[t.Tensor, tuple[t.Tensor, t.Tensor, t.Tensor]]:
        conv_y = t.relu(self.conv1(states))
        conv_y = t.relu(self.conv2(conv_y))
        conv_y = t.relu(self.conv3(conv_y))
        conv_y = t.flatten(conv_y, start_dim = 1)

        act_y = t.relu(self.act_dense1(conv_y))
        act_y = t.relu(self.act_dense2(act_y))

        act_y = self.actor(act_y)
        act_y[:, 0] = t.tanh(act_y[:, 0])
        act_y[:, 1] = t.sigmoid(act_y[:, 1])
        act_y[:, 2] = t.sigmoid(act_y[:, 2])

        if actions is None:
            return act_y

        conv_cat = t.cat((conv_y, actions), dim = 1)

        crit_y = t.relu(self.crit_dense1(conv_cat))
        crit_y = t.relu(self.crit_dense2(crit_y))

        crit_twin_y = t.relu(self.crit_twin_dense1(conv_cat))
        crit_twin_y = t.relu(self.crit_twin_dense2(crit_twin_y))

        return act_y, self.critic(crit_y), self.critic_twin(crit_twin_y)


    def actor_loss_func(self, states : t.Tensor) -> t.Tensor:
        actions = self(states)
        
        self.gradients_off(self.conv1, self.conv2, self.conv3,
                            self.crit_dense1, self.crit_dense2,
                            self.critic)
        
        _, criticism, _ = self(states, actions)
        
        self.gradients_on(self.conv1, self.conv2, self.conv3,
                            self.crit_dense1, self.crit_dense2,
                            self.critic)

        loss = -t.mean(criticism)
        
        return loss


    def critics_loss_func(self, states : t.Tensor, actions : t.Tensor, 
            targets : t.Tensor) -> t.Tensor:
        _, vals, twin_vals = self(states, actions)

        return t.mean((targets - vals) ** 2) + t.mean((targets - twin_vals) ** 2)


    def gradients_off(self, *args : t.nn.Module) -> None:
        for arg in args:
            arg.requires_grad_(False)


    def gradients_on(self, *args : t.nn.Module) -> None:
        for arg in args:
            arg.requires_grad_(True)
