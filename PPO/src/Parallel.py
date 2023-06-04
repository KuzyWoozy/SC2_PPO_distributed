import torch as t

from torch.nn.parallel import DistributedDataParallel as DDP

from src.Config import LAMBDA, PPO_CLIP, ENTROPY



def loss(func_args_dists, func_args_dists_old, actions, adv):
    # Note that we're minimizing

    actor_gain = t.tensor([0.0])
    entropy = t.tensor([0.0])
    
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


    def forward(self, agent, episode_info):
        episode_loss = t.tensor([0.0])
        G = 0.0

        for (reward, obs, func_args_dists_old, func_args_actions) in reversed(episode_info):
            func_args_dists, critic_val = agent.nn_outs(obs, func_args_actions[0])
            
            G = reward + LAMBDA * G
            ADV = G - critic_val[0]
        
            episode_loss += loss(func_args_dists, func_args_dists_old, func_args_actions, ADV)
        
        return episode_loss


class DistSyncSGD(t.nn.Module):

    def __init__(self, policy_ser):
        super().__init__()

        self.policy_dist = DDP(MonteCarlo(policy_ser), find_unused_parameters = True, gradient_as_bucket_view = True, broadcast_buffers = False)

    def forward(self, *args, **kwargs):
        return self.policy_dist.module.policy_ser(*args, **kwargs)

    def mc_loss(self, agent, episode_info):
        return self.policy_dist(agent, episode_info)

    def get_num_actions(self):
        return self.policy_dist.module.policy_ser.get_num_actions()
    
    def get_state_dict(self):
        return self.policy_dist.module.policy_ser.get_state_dict()

