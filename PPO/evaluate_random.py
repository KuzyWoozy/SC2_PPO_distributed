from pysc2.agents.random_agent import RandomAgent

from pysc2.env import sc2_env
from pysc2.env.run_loop import run_loop
from absl import app
import torch as t


def main(argv):
    agent = RandomAgent()
    
    with sc2_env.SC2Env(
        map_name="DefeatZerglingsAndBanelings",
        battle_net_map=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=[64, 64],
            feature_minimap=[64, 64],
            rgb_screen=[500, 500],
            rgb_minimap=[100, 100],
            action_space="FEATURES"),
        step_mul=8,
        visualize=True) as env:
    
        run_loop([RandomAgent()], env)
        

if __name__ == "__main__":
    app.run(main)
