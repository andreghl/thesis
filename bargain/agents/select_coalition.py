import gymnasium as gym
from gymnasium.wrappers import TimeLimit


class CoalitionSelector:

    def __init__(self, env: gym.Env,
                 path: str | None = None,
                 max_episode_steps: int = 10):

        self.env = TimeLimit(env, max_episode_steps = max_episode_steps)
        
    def load(self, path : str):
        return 0
