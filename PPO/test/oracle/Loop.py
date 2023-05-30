import time
import copy
import torch as t

from test.oracle.Checkpoint import CheckpointManager
from src.Config import LAMBDA


def run_train_loop(agent, env, max_frames = 0, episode_batch = 0):
    total_episodes = 0
    total_frames = 0


    start_time = time.time()

    obs_spec = env.observation_spec()[0]
    act_spec = env.action_spec()[0]
    agent.setup(obs_spec, act_spec)

    approx_old = copy.deepcopy(agent.approx)
    approx_old.requires_grad_(False)

    try:
        while True:
            total_episodes += 1
            
            timestep_t = env.reset()[0]

            agent.reset()

            if total_episodes % episode_batch == 1:
                approx_old = copy.deepcopy(agent.approx)
                approx_old.requires_grad_(False)


            episode_info = [] 

            while True:

                if timestep_t.last():
                    break

                total_frames += 1

                
                if agent.check_manager and agent.check_manager.time_to_save(total_frames):
                    agent.check_manager.save(total_frames, approx = agent.approx)

                if max_frames and total_frames >= max_frames:
                    return
                
                state = agent.convert_to_state(timestep_t)

                action, func_args_dists, func_args_actions, critic_val = agent.step(timestep_t, state)
                func_args_dists_old = agent.step_old(approx_old, func_args_actions[0], timestep_t, state)

                timestep_tt = env.step([action])[0]
                
                episode_info.append((timestep_tt.reward, state, critic_val, func_args_dists, func_args_dists_old, func_args_actions))

                timestep_t = timestep_tt

            
            loss = t.tensor([0.0])
            G = 0.0
            for reward, state, critic_val, func_args_dists, func_args_dists_old, func_args_actions in reversed(episode_info):

                G = reward + LAMBDA * G
                ADV = G - critic_val[0]
                
                loss += agent.approx.loss(func_args_dists, func_args_actions, func_args_dists_old, ADV)
                
            agent.optim.zero_grad()
            loss.backward()
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
                
                state = agent.convert_to_state(timestep_t)
                
                action, func_args_dists, func_args_actions, critic_val = agent.step(timestep_t, state)

                timestep_tt = env.step([action])[0]
                
                timestep_t = timestep_tt

            
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
