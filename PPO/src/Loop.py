import time
import copy
import torch as t

import torch.distributed as dist

from src.Checkpoint import CheckpointManager
from src.Config import EPOCH_BATCH, ROOT


def run_train_loop(workaround, agent, env, max_episodes):
    total_episodes = 0
    total_frames = 0

    start_time = time.time()

    obs_spec = env.observation_spec()[0]
    act_spec = env.action_spec()[0]
    agent.setup(obs_spec, act_spec)

    try:
        while True:
            if total_episodes >= max_episodes:
                return

            total_episodes += 1

            if dist.get_rank() == ROOT and agent.check_manager and agent.check_manager.time_to_save(total_episodes):
                agent.check_manager.save(total_episodes, approx = agent.approx)
            
            timestep_t = env.reset()[0]

            agent.reset()

            episode_info = []
           
            with t.no_grad():
                # Sample a trajectory
                while True:
                    if timestep_t.last():
                        break

                    total_frames += 1

                    action, func_args_dists, func_args_actions = agent.step(timestep_t)

                    timestep_tt = env.step([action])[0]
                    
                    episode_info.append((timestep_tt.reward, timestep_t, func_args_dists, func_args_actions))
                    timestep_t = timestep_tt
                
            
            for _ in range(EPOCH_BATCH):
                # Could do an optimization here to process the first iteration faster
                # by avoiding recalculation of dists

                agent.optim.zero_grad()
                workaround(episode_info).backward()
                agent.optim.step()

                
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))



def run_evaluate_loop(agent, env):
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
