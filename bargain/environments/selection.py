from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from .utils import Observation, sample, is_truncated
from typing import Any

import gymnasium as gym
import numpy as np


class Selection(gym.Env):

    def __init__(self,
                 obs: Observation):

        self.obs = obs
        self.obs.active = "selection"
        self.action_space = MultiBinary(obs.n_vehicles)
        self.observation_space = Dict({
            "instance": Box(
                low = -np.inf,
                high = np.inf,
                # the instance matrix contains one depot and has 4 columns
                shape = (1, (obs.n_customers + 1) * 4),
                dtype = obs.float
            ),
            "coalition": MultiBinary(obs.n_vehicles),
            "proposal": Box(
                low = 0,
                high = 1,
                shape = (obs.n_vehicles, ),
                dtype = obs.float
            ),
            "response": MultiBinary(obs.n_vehicles),
            "time": Discrete(obs.max_time, dtype = obs.int),
            "proposer": Discrete(obs.n_vehicles, dtype = obs.int),
            "agent": Discrete(obs.n_vehicles, dtype = obs.int),
            "role": Discrete(3, dtype = obs.int)}
        )

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed = seed, options = options)
        obs = self.obs.reset()

        return obs, {}

    def step(self, action: np.ndarray):
        coalition = sample(action)
        self.obs["coalition"] = coalition.copy()
        obs = self.obs.step()
        reward, terminated = self.reward(coalition)
        truncated = is_truncated(self.obs["time"])

        return obs, reward, terminated, truncated, {}

    def reward(self, coalition: np.ndarray):

        agent = self.obs["agent"]

        while self.obs["role"] != 0:

            while self.obs["role"] == 1:
                self.obs.propose(self.obs)
                self.obs.step()


            while self.obs["role"] == 2:
                self.obs.respond(self.obs)
                self.obs.step()

        proposal = self.obs["proposal"]
        response = self.obs["response"]

        value = self.obs.value(coalition)
        done = np.all(coalition == response)
        reward = coalition[agent] * proposal[agent] * response[agent] * value

        self.obs.internal_reset()
        return float(reward), bool(done)

class _Selection(Selection):

    def __init__(self,
                 obs: Observation):
        super().__init__(obs)
        self.action_space = Box(
            low = 0,
            high = 1,
            shape = (obs.n_vehicles, ),
            dtype = obs.float
        )

class SelectionDummy(_Selection):

    def reward(self, coalition: np.ndarray):

        if np.sum(coalition) <= 1:
            return -0.1, True

        agent = self.obs["proposer"]
        if coalition[agent] != 1:
            return -0.1, True

        value = self.obs.value(coalition)
        n = np.sum(coalition) if np.sum(coalition) > 0 else 1

        while self.obs['role'] != 0:
            self.obs.step()

        self.obs.internal_reset()


        return (1 / n) * coalition[agent] * value, False

if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env
    from gymnasium.envs.registration import registry

    if "Selection_Discrete-v0" not in registry:
        gym.register(
            id = "Selection_Discrete-v0",
            entry_point = Selection,
            max_episode_steps = 10)

    if "Selection_Continuous-v1" not in registry:
        gym.register(
            id = "Selection_Continuous-v1",
            entry_point = _Selection,
            max_episode_steps = 10)

    if "Selection_Continuous-v0" not in registry:
        gym.register(
            id = "Selection_Continuous-v0",
            entry_point = SelectionDummy,
            max_episode_steps = 10)


    vehicles = 3
    customers = 9
    maxtime = 10
    observation = Observation(vehicles, customers, maxtime)

    # Discrete action space environment
    env = gym.make(id = "Selection_Discrete-v0",
                   obs = observation)

    try:
        check_env(env,
                  warn = False,
                  skip_render_check = True); print("Passed!")

    except Exception as e:
        print(f"Failed!\n")
        print(e)

    # Continuous action space environment
    env = gym.make(id = "Selection_Continuous-v0",
                   obs = observation)

    try:
        check_env(env,
                  warn = False,
                  skip_render_check = True); print("Passed!")

    except Exception as e:
        print(f"Failed!\n")
        print(e)

    # Continuous action space environment
    env = gym.make(id = "Selection_Continuous-v1",
                   obs = observation)

    try:
        check_env(env,
                  warn = False,
                  skip_render_check = True); print("Passed!")

    except Exception as e:
        print(f"Failed!\n")
        print(e)