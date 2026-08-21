from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from .utils import Observation, sample, is_truncated
from typing import Any

import gymnasium as gym
import numpy as np


class Response(gym.Env):

    def __init__(self,
                 obs: Observation):

        self.obs = obs
        self.obs.active = "response"
        self.action_space = MultiBinary(obs.n_vehicles)
        self.observation_space = Dict({
            "instance": Box(
                low = -np.inf,
                high = np.inf,
                # the instance matrix contains one depot and has 4 columns
                shape = (1, (obs.n_customers + 1) * 4),
                dtype = obs.float
            ),
            "coalition": Box(
                low = 0,
                high = 1,
                shape = (obs.n_vehicles, ),
                dtype = obs.float
            ),
            "proposal": Box(
                low = 0,
                high = 1,
                shape = (obs.n_vehicles, ),
                dtype = obs.float
            ),
            "response": Box(
                low = 0,
                high = 1,
                shape = (obs.n_vehicles, ),
                dtype = obs.float
            ),
            "time": Discrete(obs.max_time, dtype = obs.int),
            "proposer": Discrete(obs.n_vehicles, dtype = obs.int),
            "agent": Discrete(obs.n_vehicles, dtype = obs.int),
            "role": gym.spaces.Discrete(3, dtype = obs.int)}
        )

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed = seed, options = options)
        obs = self.obs.reset()

        return obs, {}

    def step(self, action: np.ndarray):
        response = sample(action)
        self.obs["response"] = response.copy()
        obs = self.obs.step()
        reward, terminated = self.reward(response)
        truncated = is_truncated(self.obs["time"])
        # print("step", action, reward, coalition)

        return obs, reward, terminated, truncated, {}

    def reward(self, response: np.ndarray):

        agent = self.obs["agent"]
        while self.obs["role"] == 2:
            self.obs.respond(self.obs)
            self.obs.step()

        coalition = self.obs["coalition"]
        proposal = self.obs["proposal"]
        value = self.obs.value(coalition)
        done = np.all(coalition == response)
        reward = coalition[agent] * proposal[agent] * response[agent] * value
        # print("reward", reward, coalition, proposal, response)

        """
        is_proposing = agent == proposer
        if is_proposing and response[agent] == 0:
            return -1.0, bool(done)
        """
        self.obs.internal_reset()
        return float(reward * (int(done) + 0.2)), bool(done)

class _Response(Response):

    def __init__(self,
                 obs: Observation):
        super().__init__(obs)
        self.action_space = Box(
            low = 0,
            high = 1,
            shape = (obs.n_vehicles, ),
            dtype = obs.float
        )


if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env
    from gymnasium.envs.registration import registry

    if "Response_Discrete-v0" not in registry:
        gym.register(
            id = "Response_Discrete-v0",
            entry_point = Response,
            max_episode_steps = 10)

    if "Response_Continuous-v0" not in registry:
        gym.register(
            id = "Response_Continuous-v0",
            entry_point = _Response,
            max_episode_steps = 10)


    vehicles = 3
    customers = 9
    maxtime = 10
    observation = Observation(vehicles, customers, maxtime)

    # Discrete action space environment
    env = gym.make(id = "Response_Discrete-v0",
                   obs = observation)

    try:
        check_env(env,
                  warn = False,
                  skip_render_check = True); print("Passed!")

    except Exception as e:
        print(f"Failed!\n")
        print(e)

    # Continuous action space environment
    env = gym.make(id = "Response_Continuous-v0",
                   obs = observation)

    try:
        check_env(env,
                  warn = False,
                  skip_render_check = True); print("Passed!")

    except Exception as e:
        print(f"Failed!\n")
        print(e)