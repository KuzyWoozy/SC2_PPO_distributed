import gym

import torch as t
from typing import Any

import src

from train import ENV, HIDDEN_SIZE, FRAME_SKIP, RaceEnvi


# SETTINGS
CHECKPOINT_PATH = "checkpoints/" + ENV + "-" + "850000.chkpt"

def evaluate(env : src.envi.Envi, model : t.nn.Module) -> None:
    model.eval()

    while True:
        state_tt, _ = env.reset()

        while True:
            state_t = state_tt

            action = model(state_t.unsqueeze(0)).detach()
            #action = env.random_action()
            print(action)
            
            state_t = t.roll(state_t, shifts = -FRAME_SKIP, dims = 0)
            for skip in range(FRAME_SKIP):
                state, reward, terminal, trunc, _ = env.step(action)
                state_t[skip + 1] = state

            state, reward, terminal, trunc, _ = env.step(action)
            state_tt = t.roll(state_t, shifts = -1, dims = 0)
            state_tt[FRAME_SKIP] = state

            if terminal:
                break

# MAIN
def main() -> None:
    
    env = RaceEnvi(FRAME_SKIP, render_mode = "human")

    model = src.net.ActorCriticTwin(FRAME_SKIP, HIDDEN_SIZE, env.action_elems())
    
    chk = t.load(CHECKPOINT_PATH)
    model.load_state_dict(chk["model"])

    evaluate(env, model)

if __name__ == "__main__":
    main()
