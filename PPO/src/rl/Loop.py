import time, copy, sys
import torch as t

import torch.distributed as dist

from src.Config import MAX_AGENT_STEPS, ROOT, EPOCH_BATCH


def train_loop(agent, env):
    total_agent_steps = 0

    start_time = time.time()

    obs_spec = env.observation_spec()[0]
    act_spec = env.action_spec()[0]
    agent.setup(obs_spec, act_spec)

    try:
        while True:

            if total_agent_steps >= MAX_AGENT_STEPS:
                agent.save(total_agent_steps)
                dist.destroy_process_group()
            
            timestep_t = env.reset()[0]

            agent.reset()

            episode_info = []
            shortcut = []
           
            # Sample a trajectory
            while True:
                if timestep_t.last():
                    break

                action, func_args_dists, func_args_actions, crit = agent.step(timestep_t)

                timestep_tt = env.step([action])[0] 
                
                total_agent_steps += 1

                agent.save_if_rdy(total_agent_steps)

                episode_info.append((timestep_tt.reward, timestep_t, [i.detach() for i in func_args_dists], func_args_actions))

                shortcut.append((func_args_dists, crit))

                timestep_t = timestep_tt

            # first step optimization
            agent.optim.zero_grad()
            agent.policy.mc_loss(agent, episode_info, shortcut).backward()
            agent.optim.step()

            for _ in range(EPOCH_BATCH - 1):
                agent.optim.zero_grad()
                agent.policy.mc_loss(agent, episode_info).backward()
                agent.optim.step()

            
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps" % (
            elapsed_time, total_agent_steps))



def evaluate_loop(agent, env):
    total_episodes = 0
    total_frames = 0

    start_time = time.time()

    obs_spec = env.observation_spec()[0]
    act_spec = env.action_spec()[0]
    
    agent.setup(obs_spec, act_spec)

    try:
        while True:
            total_episodes += 1
            
            timestep_t = env.reset()[0]

            agent.reset()

            while True:

                if timestep_t.last():
                    break

                total_frames += 1
                
                action, func_args_dists, func_args_actions = agent.step(timestep_t)

                timestep_tt = env.step([action])[0]
                
                timestep_t = timestep_tt

            
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
