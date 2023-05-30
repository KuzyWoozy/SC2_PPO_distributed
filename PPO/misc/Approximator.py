import torch as t

"""
class Convolve(t.nn.Module):
    def __init__(self):
        

    def forward(self):

class Actor(t.nn.Module):


class Critic(t.nn.Module):
    def __init__(self)


class ActorCritic(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        

    
    def convolve(self, states : t.Tensor) -> t.Tensor:
        


    def actor_forward(self, states : t.Tensor) -> t.Tensor:
        


    def critic_forward(self, states, actions):
        c_y = t.cat((conv_y, actions), dim = 1)
        c_y = t.relu(self.critic_dense1(c_y))
        c_y = t.relu(self.critic_dense2(c_y))

        return self.critic_output(c_y)


    def forward(self, states : t.Tensor): 
        conv_y = self.convolve(states)

        return self.actor_forward(conv_y)


    def actor_loss(self, states : t.Tensor) -> t.Tensor:
        conv_y = self.convolve(states)

        actions = self.actor_forward(conv_y)

        criticism = self.critic_forward(conv_y, actions)

        return -t.mean(criticism, dim = 1)


    def critics_loss(self, states : t.Tensor, actions : t.Tensor, targets : t.Tensor) -> t.Tensor:
        conv_y = self.convolve(states)
        
        return t.mean((targets - self.critic_forward(conv_y, actions)) ** 2, dim = 1)
"""

