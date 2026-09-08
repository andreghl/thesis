from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from bargain import generate_observation
from bargain.env.dummies import *
from typing import Any

import gymnasium as gym
import numpy as np


class Proposer(gym.Env):

    _INT = np.int64
    _BIN = np.int8
    _BOOL = np.bool
    _FLOAT = np.float32

    def __init__(self,
                 n_vehicles: int,
                 n_customers: int,
                 max_time: int = 10,
                 agent_path: str | None = None):

        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.max_time = max_time
        self.agent_path = agent_path
        self.action_dim = 2 * n_vehicles + 1

        self.action_space = Box(low = 0,
                                high = 1,
                                shape = (self.action_dim, ),
                                dtype = self._FLOAT)

        self.observation_space = Dict({
            "instance": Box(low = - np.inf,
                            high = np.inf,
                            shape = (1, (self.n_customers + 1) * 4),
                            dtype = self._FLOAT),
            "coalition": MultiBinary(self.n_vehicles),
            "proposal": Box(low = 0,
                            high = 1,
                            shape = (self.n_vehicles, ),
                            dtype = self._FLOAT),
            "response": MultiBinary(self.n_vehicles),
            "time": Discrete(self.max_time),
            "proposer": Discrete(self.n_vehicles),
            "agent": Discrete(self.n_vehicles),
            "role": Discrete(3)
        })

        # utilities
        self.obs = None
        self.instance = None
        self.routes = None
        self.coalitions = None
        self.gains = None
        self.shapley = None
        self.nucleolus = None
        self.order = None

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None):

        super().reset(seed = seed, options = options)
        np.random.seed(seed)

        (self.instance,
         self.routes,
         self.coalitions,
         self.gains,
         self.shapley,
         self.nucleolus) = generate_observation(self.n_vehicles,
                                                self.n_customers,
                                                radius = 1.0)

        order = np.random.choice(np.arange(self.n_vehicles),
                                 size = self.n_vehicles,
                                 replace = False)
        self.order = list(order)

        proposer = self.order.pop()
        coalition = coalition_heuristic(self.coalitions, self.gains, dtype = self._BIN)
        proposal = proposal_heuristic(self.n_vehicles, coalition, dtype = self._FLOAT)
        response = response_heuristic(coalition, proposal, dtype = self._BIN)
        time = self.max_time - 1
        role = 0

        obs = {"instance": flatten(self.instance[:, 1:], dtype = self._FLOAT),
               "coalition": coalition,
               "proposal": proposal,
               "response": response,
               "time": self._INT(time),
               "proposer": proposer,
               "agent": proposer,
               "role": self._INT(role)}

        self.obs = obs

        return self.obs, {}

    def step(self, action: np.ndarray):
        _mask = mask(action, self.obs["role"], self.n_vehicles)
        self.obs["coalition"], self.obs["proposal"] = get_actions_proposer(action,
                                                   self.n_vehicles,
                                                   (self._BIN, self._FLOAT))
        self.obs["response"] = response(self.obs, self.order, self.n_vehicles, self.agent_path)
        reward = self.reward()
        terminated = self.terminated()
        truncated = False

        order = np.random.choice(np.arange(self.n_vehicles),
                                 size = self.n_vehicles,
                                 replace = False)
        self.order = list(order)

        proposer = self.order.pop()

        self.obs["time"] -= 1
        self.obs["proposer"] = proposer
        self.obs["agent"] = proposer

        # print("DEBUG", reward, action)

        return self.obs, reward, terminated, truncated, {}


    def gain(self, coalition: np.ndarray):
        mask = np.all(coalition == self.coalitions, axis = 1)
        if not np.any(mask):
            return 0.0
        else:
            return self.gains[mask].astype(self._FLOAT).item()

    def reward(self):

        coalition = self.obs["coalition"]
        proposal = self.obs["proposal"]
        response = self.obs["response"]
        value = self.gain(coalition)
        i = self.obs["agent"]
        n = np.maximum(np.sum(coalition), 1)

        _reward = np.sum(coalition * proposal * response)
        reward = ((1/n) - np.max(proposal)) * 0.1
        if np.isclose(reward, 0.0):
            reward = _reward

        if np.sum(coalition) < 2:
            reward = -(2 - np.sum(coalition)) * np.abs(reward)

        return np.clip(reward, -1.0, 1.0)

    def terminated(self):

        coalition = self.obs["coalition"]
        response = self.obs["response"]
        agent = self.obs["agent"]
        _terminated = np.all(coalition == (response * coalition))

        return bool(_terminated)

"""
if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env
    from gymnasium.envs.registration import registry, register
    import gymnasium as gym

    name = "Proposer-v0"
    if name not in registry:
        register(id = name,
                 entry_point = Proposer,
                 max_episode_steps = 90)

    env = gym.make(id = name,
                   n_vehicles = 3,
                   n_customers = 9)

    try:
        check_env(env,
                  warn = False,
                  skip_render_check = True); print("Passed!")

    except Exception as e:
        print(f"Failed!\n")
        print(e)
"""