import time
import torch as t

import torch.distributed as dist

from src.Config import MAX_AGENT_STEPS, ROOT, EPOCH_BATCH, SYNC, DTYPE, TIMING_EPISODE_DELAY




def train_loop(agent, env):
    total_agent_steps = 0
    episode = 0
    
    obs_spec = env.observation_spec()[0]
    act_spec = env.action_spec()[0]
    agent.setup(obs_spec, act_spec)

    try:
        while True:

            if episode == TIMING_EPISODE_DELAY:
                start_time = time.time()

            if total_agent_steps >= MAX_AGENT_STEPS:
                if SYNC:
                    if dist.get_rank() == ROOT:
                        agent.save(total_agent_steps)
                    dist.destroy_process_group()
                else:
                    agent.save(total_agent_steps)
                break
                

            timestep_t = env.reset()[0]

            agent.reset()

            episode_info = []
            shortcut = []
           
            # Sample a trajectory
            while True:
                if timestep_t.last():
                    break

                action, func_args_dists, func_args_actions, mask, crit = agent.step(timestep_t)

                timestep_tt = env.step([action])[0] 

                total_agent_steps += 1
                
                if SYNC:
                    if dist.get_rank() == ROOT:
                        agent.save_if_rdy(total_agent_steps)
                else:
                    agent.save_if_rdy(total_agent_steps)

                episode_info.append((t.tensor([timestep_tt.reward], dtype = DTYPE, device = agent.policy.device), agent.obs_to_state(timestep_t), mask, [i.detach() for i in func_args_dists], func_args_actions))

                shortcut.append((func_args_dists, crit))

                timestep_t = timestep_tt

            """
            if SYNC:
                with agent.policy.policy_dist.no_sync():
                    agent.optim.zero_grad()
                    agent.policy.mc_loss(agent, episode_info, shortcut).backward()
                    agent.optim.step()

                    for _ in range(EPOCH_BATCH - 2):
                        agent.optim.zero_grad()
                        agent.policy.mc_loss(agent, episode_info).backward()
                        agent.optim.step()

                agent.optim.zero_grad()
                agent.policy.mc_loss(agent, episode_info).backward()
                agent.optim.step()
            """
           
            
            agent.policy.mc_loss(agent, episode_info, shortcut).backward()
            agent.optim.step()
            agent.optim.zero_grad()
            for _ in range(EPOCH_BATCH - 1):
                agent.policy.mc_loss(agent, episode_info).backward()
                agent.optim.step()
                agent.optim.zero_grad()
            
            episode += 1 # Completed
            
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
                
                action, _, _, _ = agent.step(timestep_t)

                timestep_tt = env.step([action])[0]
                
                timestep_t = timestep_tt

            
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
