import time
import torch as t

import torch.distributed as dist

from src.Parallel import SerialSGD
from src.Config import MAX_AGENT_STEPS, ROOT, SYNC, DTYPE, TIMING_EPISODE_DELAY, TRAJ, PROFILE


def network_update(agent, episode_info, terminate):
     
    agent.optim.zero_grad()

    if terminate:
        bootstrap = t.tensor([0.0], dtype = DTYPE, device = agent.policy.device)
        agent.policy.mc_loss(agent, episode_info, bootstrap).backward()
    else:
        _, _, _, _, (bootstrap,) = episode_info[-1]
        agent.policy.mc_loss(agent, episode_info[:-1], bootstrap).backward()
        
    agent.old_policy = agent.policy.freeze()
    agent.optim.step()
    


def evaluate_loop(agent, env, epi = 25):
    steps = 0
    episodes = 0
    score = 0

    obs_spec, = env.observation_spec()
    act_spec, = env.action_spec()
    agent.setup(obs_spec, act_spec)

    start_timer = time.time()

    try:
        with t.no_grad():
            for _ in range(epi):
                timestep_t, = env.reset()
                episode_steps = 0
                
                agent.reset()

                reward = 0

                # Sample a trajectory
                while True:
                   
                    action, func_args_dists, func_args_dists_old, func_args_actions, crit = agent.step(timestep_t)
                    
                    timestep_tt, = env.step([action])
                    
                    reward += timestep_tt.reward

                    episode_steps += 1           
                    
                    if timestep_tt.last():
                        score += reward
                        break

                    timestep_t = timestep_tt
                    
                    steps += 1
                
                episodes += 1 # Completed
            
            return score / episodes

    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_timer
        
        print("Took %.3f seconds for %s steps" % (
            elapsed_time, steps))




def train_loop(agent, env):
    
    steps = 0
    episodes = 0
    
    obs_spec, = env.observation_spec()
    act_spec, = env.action_spec()
    agent.setup(obs_spec, act_spec)

    start_timer = time.time()

    try:
        while True:

            if episodes == TIMING_EPISODE_DELAY:
                start_timer = time.time()
            
            timestep_t, = env.reset()
            episode_steps = 0
            
            agent.reset()

            episode_info = []
           
            # Sample a trajectory
            while True:
               
                action, func_args_dists, func_args_dists_old, func_args_actions, crit = agent.step(timestep_t)

                timestep_tt, = env.step([action])
                
                if SYNC:
                    if dist.get_rank() == ROOT:
                        agent.save_if_rdy(steps)
                else:
                    agent.save_if_rdy(steps)

                episode_info.append((t.tensor([timestep_tt.reward], dtype = DTYPE, device = agent.policy.device), func_args_dists, func_args_dists_old, func_args_actions, crit))

                episode_steps += 1           
                if (episode_steps % TRAJ == 0) or (terminate := timestep_tt.last()):                    

                    network_update(agent, episode_info, terminate) 

                    episode_info = []
                   
                    if PROFILE and SYNC:
                        non_reduced_steps = steps
                        steps_tensor = t.tensor([steps], dtype = t.int32)
                        dist.all_reduce(steps_tensor)
                        steps = steps_tensor.item() 

                    if steps >= MAX_AGENT_STEPS:
                        if SYNC:
                            if dist.get_rank() == ROOT:
                                agent.save(steps)
                        else:
                            agent.save(steps)
                        return
                    
                    if PROFILE and SYNC:
                        steps = non_reduced_steps
                    
                    if terminate:
                        break
                    
                timestep_t = timestep_tt
                
                steps += 1
            
            episodes += 1 # Completed
            
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_timer
        
        if SYNC:
            if PROFILE:
                if dist.get_rank() == ROOT:
                    with open("output.txt", "a+") as f:
                        agent.policy = SerialSGD(agent.policy.policy.module.policy_ser, t.device("cpu"))
                        f.write(f"{dist.get_world_size()}, {elapsed_time}, {evaluate_loop(agent, env)}\n")  
        

        print("Took %.3f seconds for %s steps" % (
            elapsed_time, steps))



