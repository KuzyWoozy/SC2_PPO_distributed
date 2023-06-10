import torch as t

from torch.nn.parallel import DistributedDataParallel as DDP

from src.Config import LAMBDA, PPO_CLIP, ENTROPY, DTYPE, GPU




class MonteCarlo(t.nn.Module):

    def __init__(self, policy_ser, device):
        super().__init__()
        self.policy_ser = policy_ser
        self.device = device
    
    def loss(self, func_args_dists, func_args_dists_old, actions, adv):
        # Note that we're minimizing

        actor_gain = t.tensor([0.0], dtype = DTYPE, device = self.device)
        entropy = t.tensor([0.0], dtype = DTYPE, device = self.device)
        
        adv_detached = adv.detach()

        for out, out_old, action in zip(func_args_dists, func_args_dists_old, actions):
            actor_gain -= t.min((out[:, action] / out_old[:, action]) * adv_detached, t.clip(out[:, action] / out_old[:, action], min = 1 - PPO_CLIP, max = 1 + PPO_CLIP) * adv_detached)
            # Trick to avoid having to avoid conditional
            entropy -= t.sum(out * t.log(out + 1e-8))

        critic_loss = adv ** 2

        return actor_gain + critic_loss + (ENTROPY * entropy)



    def forward(self, agent, episode_info, shortcut):
        episode_loss = t.tensor([0.0], dtype = DTYPE, device = self.device)
        G = t.tensor([0.0], dtype = DTYPE, device = self.device)

        if shortcut:
            shortcut_length = len(shortcut)

        for i, (reward, obs, func_args_dists_old, func_args_actions) in enumerate(reversed(episode_info)):
            if shortcut:
                func_args_dists, critic_val = shortcut[shortcut_length - 1 - i]
            else:
                func_args_dists, critic_val = agent.nn_outs(obs, func_args_actions[0])

            G = reward + LAMBDA * G
            ADV = G - critic_val[0]
       
            episode_loss += self.loss(func_args_dists, func_args_dists_old, func_args_actions, ADV)
        
        return episode_loss


class SerialSGD(t.nn.Module):

    def __init__(self, policy_ser, device):
        super().__init__()
        
        if GPU:
            policy_ser = policy_ser.to(dtype = DTYPE, device = device)
            policy_ser = t.cuda.make_graphed_callables(policy_ser, (t.randn((1, 12, 64, 64), dtype = DTYPE, device = device),))

        self.policy = MonteCarlo(policy_ser, device)
        self.device = device

    def forward(self, *args, **kwargs):
        return self.policy.policy_ser(*args, **kwargs)

    def mc_loss(self, agent, episode_info, shortcut = None):
        return self.policy(agent, episode_info, shortcut)

    def get_state_dict(self):
        return self.policy.policy_ser.state_dict()
    

class DistSyncSGD(t.nn.Module):

    def __init__(self, policy_ser, device):
        super().__init__()

        mc = MonteCarlo(policy_ser)

        mc = mc.to(dtype = DTYPE, device = device)
        
        if GPU:
            self.policy_dist = DDP(mc, find_unused_parameters = True, gradient_as_bucket_view = True, broadcast_buffers = False, device_ids = [policy_ser.get_device()])
        else:
            self.policy_dist = DDP(mc, find_unused_parameters = True, gradient_as_bucket_view = True, broadcast_buffers = False)

        self.device = device


    def forward(self, *args, **kwargs):
        return self.policy_dist.module.policy_ser(*args, **kwargs)

    def mc_loss(self, agent, episode_info, shortcut = None):
        return self.policy_dist(agent, episode_info, shortcut)

    def get_num_actions(self):
        return self.policy_dist.module.policy_ser.get_num_actions()
    
    def get_state_dict(self):
        return self.policy_dist.module.policy_ser.get_state_dict()

    def get_device(self):
        return self.device
