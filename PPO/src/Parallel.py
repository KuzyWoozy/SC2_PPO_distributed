import torch as t

import copy

from torch.nn.parallel import DistributedDataParallel as DDP

from src.Config import GAMMA, PPO_CLIP, DTYPE, GPU, ENTROPY, VALUE_COEFF



class MonteCarlo(t.nn.Module):

    def __init__(self, policy_ser, device):
        super().__init__()
        self.policy_ser = policy_ser
        self.device = device

    def loss(self, actor_gain, critic_loss, entropy, func_args_dists, func_args_dists_old, actions, adv):
        # Note that we're minimizing

        adv_detached = adv.detach()

        for out, out_old, action in zip(func_args_dists, func_args_dists_old, actions):
            print((out[:, action] / out_old[:, action]))
            print("\n\n\n")
            actor_gain += t.min((out[:, action] / out_old[:, action]) * adv_detached, t.clip(out[:, action] / out_old[:, action], min = 1 - PPO_CLIP, max = 1 + PPO_CLIP) * adv_detached)          
            # Trick to avoid having to avoid conditional
            entropy -= t.sum(out * t.log(out + 1e-8))
            
        critic_loss += (adv ** 2)
        
    
    def forward(self, agent, episode_info, bootstrap):
        actor_gain = t.tensor([0.0], dtype = DTYPE, device = self.device)
        critic_loss = t.tensor([0.0], dtype = DTYPE, device = self.device)
        entropy = t.tensor([0.0], dtype = DTYPE, device = self.device)

        episode_length = len(episode_info)

        G = bootstrap.detach()

        for (reward, func_args_dists, func_args_dists_old, func_args_actions, critic_val) in reversed(episode_info): 
            G = reward + GAMMA * G
            ADV = G - critic_val[0]
       
            self.loss(actor_gain, critic_loss, entropy, func_args_dists, func_args_dists_old, func_args_actions, ADV)
        
        return ((-actor_gain) + (VALUE_COEFF * critic_loss) - (ENTROPY * entropy)) / episode_length


class SerialSGD(t.nn.Module):

    def __init__(self, policy_ser, device):
        super().__init__()
        
        policy_ser = policy_ser.to(dtype = DTYPE, device = device)

        #if GPU:
        #    policy_ser = t.cuda.make_graphed_callables(policy_ser, (t.randn((1, 5, 64, 64), dtype = DTYPE, device = device),))

        self.policy = MonteCarlo(policy_ser, device)
        self.device = device

    def forward(self, *args, **kwargs):
        return self.policy.policy_ser(*args, **kwargs)

    def mc_loss(self, agent, episode_info, bootstrap):
        return self.policy(agent, episode_info, bootstrap)

    def get_state_dict(self):
        return self.policy.policy_ser.state_dict()
   
    def sample_args(self, *args, **kwargs):
        return self.policy.policy_ser.sample_args(*args, **kwargs)
    
    def probs_args(self, *args, **kwargs):
        return self.policy.policy_ser.probs_args(*args, **kwargs)
    
    def freeze(self):
        return copy.deepcopy(self.policy.policy_ser).requires_grad_(False).to(self.device)


class DistSyncSGD(t.nn.Module):

    def __init__(self, policy_ser, device):
        super().__init__()
        
        policy_ser = policy_ser.to(dtype = DTYPE, device = device)

        if GPU:
            self.policy = DDP(MonteCarlo(policy_ser, device), find_unused_parameters = True, gradient_as_bucket_view = True, broadcast_buffers = False, device_ids = [device])
        else:
            self.policy = DDP(MonteCarlo(policy_ser, device), find_unused_parameters = True, gradient_as_bucket_view = True, broadcast_buffers = False)
        
        self.device = device


    def forward(self, *args, **kwargs):
        return self.policy.module.policy_ser(*args, **kwargs)

    def mc_loss(self, agent, episode_info, bootstrap):
        return self.policy(agent, episode_info, bootstrap)

    def get_state_dict(self):
        return self.policy.module.policy_ser.state_dict()

    def sample_args(self, *args, **kwargs):
        return self.policy.module.policy_ser.sample_args(*args, **kwargs)
    
    def probs_args(self, *args, **kwargs):
        return self.policy.module.policy_ser.probs_args(*args, **kwargs)

    def freeze(self):
        return copy.deepcopy(self.policy.module.policy_ser).requires_grad_(False).to(self.device)
