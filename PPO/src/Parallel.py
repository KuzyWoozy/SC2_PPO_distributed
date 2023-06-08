import torch as t

from torch.nn.parallel import DistributedDataParallel as DDP

from src.Config import LAMBDA, PPO_CLIP, ENTROPY, DTYPE, GPU



def loss(func_args_dists, func_args_dists_old, actions, adv, device):
    # Note that we're minimizing

    actor_gain = t.tensor([0.0], dtype = DTYPE, device = device)
    entropy = t.tensor([0.0], dtype = DTYPE, device = device)
    
    adv_detached = adv.detach()


    for out, out_old, action in zip(func_args_dists, func_args_dists_old, actions):
        actor_gain -= t.min((out[:, action] / out_old[:, action]) * adv_detached, t.clip(out[:, action] / out_old[:, action], min = 1 - PPO_CLIP, max = 1 + PPO_CLIP) * adv_detached)

        out_pos = out[out > 0.0]
        entropy -= t.sum(out_pos * t.log(out_pos), axis = 0)

    critic_loss = adv ** 2

    return actor_gain + critic_loss + (ENTROPY * entropy)


class MonteCarlo(t.nn.Module):

    def __init__(self, policy_ser):
        super().__init__()
        self.policy_ser = policy_ser


    def forward(self, agent, episode_info, shortcut):
        episode_loss = t.tensor([0.0], dtype = DTYPE, device = self.policy_ser.get_device())
        G = t.tensor([0.0], dtype = DTYPE, device = self.policy_ser.get_device())

        if shortcut:
            shortcut_length = len(shortcut)

        for i, (reward, obs, func_args_dists_old, func_args_actions) in enumerate(reversed(episode_info)):
            if shortcut:
                func_args_dists, critic_val = shortcut[shortcut_length - 1 - i]
            else:
                func_args_dists, critic_val = agent.nn_outs(obs, func_args_actions[0])

            G = reward + LAMBDA * G
            ADV = G - critic_val[0]
       
            episode_loss += loss(func_args_dists, func_args_dists_old, func_args_actions, ADV, self.policy_ser.get_device())
        
        return episode_loss


class SerialSGD(t.nn.Module):

    def __init__(self, policy_ser):
        super().__init__()

        mc = MonteCarlo(policy_ser)

        mc.to(device = policy_ser.get_device(), dtype = DTYPE)

        self.policy = mc

    def forward(self, *args, **kwargs):
        return self.policy.policy_ser(*args, **kwargs)

    def mc_loss(self, agent, episode_info, shortcut = None):
        return self.policy(agent, episode_info, shortcut)

    def get_num_actions(self):
        return self.policy.policy_ser.get_num_actions()
    
    def get_state_dict(self):
        return self.policy.policy_ser.get_state_dict()
    
    def get_device(self):
        return self.policy.policy_ser.get_device()



class DistSyncSGD(t.nn.Module):

    def __init__(self, policy_ser):
        super().__init__()

        mc = MonteCarlo(policy_ser)

        mc.to(device = policy_ser.get_device(), dtype = DTYPE)

        if GPU:
            self.policy_dist = DDP(mc, find_unused_parameters = True, gradient_as_bucket_view = True, broadcast_buffers = False, device_ids = [policy_ser.get_device()])
        else:
            self.policy_dist = DDP(mc, find_unused_parameters = True, gradient_as_bucket_view = True, broadcast_buffers = False)



    def forward(self, *args, **kwargs):
        return self.policy_dist.module.policy_ser(*args, **kwargs)

    def mc_loss(self, agent, episode_info, shortcut = None):
        return self.policy_dist(agent, episode_info, shortcut)

    def get_num_actions(self):
        return self.policy_dist.module.policy_ser.get_num_actions()
    
    def get_state_dict(self):
        return self.policy_dist.module.policy_ser.get_state_dict()

    def get_device(self):
        return self.policy_dist.module.policy_ser.get_device()
