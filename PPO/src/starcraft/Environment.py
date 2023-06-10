from pysc2.env import sc2_env

from src.Config import MINIGAME_NAME, SEED


class StarcraftMinigame(sc2_env.SC2Env):

    def __init__(self, agent, viz = False):
        if viz:
            super().__init__(map_name = MINIGAME_NAME,
                battle_net_map = False,
                players = [sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=sc2_env.parse_agent_interface_format(
                    rgb_screen = [500, 500],
                    rgb_minimap = [100, 100],
                    feature_screen = [64, 64],
                    feature_minimap = [64, 64],
                    action_space = "FEATURES"),
                visualize = True,
                step_mul = 8,
                random_seed = SEED)
        
        else:
            super().__init__(map_name = MINIGAME_NAME,
                battle_net_map = False,
                players = [sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=sc2_env.parse_agent_interface_format(
                    feature_screen = [64, 64],
                    feature_minimap = [64, 64]),
                visualize = False,
                realtime = False,
                step_mul = 8,
                random_seed = SEED)
