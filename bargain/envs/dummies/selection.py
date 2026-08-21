from bargain.envs.dummies import coalition_heuristic, proposal_heuristic, response_heuristic
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from bargain.envs.dummies import select, sample, flatten
from bargain import generate_observation
from typing import Any
import gymnasium as gym
import numpy as np


class Selection(gym.Env):

    _INT = np.int64
    _BIN = np.int8
    _BOOL = np.bool
    _FLOAT = np.float32

    def __init__(self,
                 n_vehicles: int,
                 n_customers: int,
                 max_time: int = 10):

        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.max_time = max_time
        self.max_steps = n_vehicles + 2
        self.timestep = 0

        # self.action_space = MultiBinary(self.n_vehicles)
        self.action_space = Box(low = 0,
                                high = 1,
                                shape = (self.n_vehicles, ),
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

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed = seed, options = options)
        np.random.seed(seed)
        self.timestep = 0

        (self.instance,
         self.routes,
         self.coalitions,
         self.gains,
         self.shapley,
         self.nucleolus) = generate_observation(self.n_vehicles,
                                                self.n_customers,
                                                radius = 1.0)

        proposer = select(self.n_vehicles, dtype = self._INT)
        coalition = coalition_heuristic(self.coalitions, dtype = self._BIN)
        proposal = proposal_heuristic(self.n_vehicles, coalition, dtype = self._FLOAT)
        response = response_heuristic(coalition, proposal, dtype = self._BIN)
        time = select(self.max_time - 1, dtype = self._INT)
        role = 0

        obs = {"instance": flatten(self.instance[:, 1:], dtype = self._FLOAT),
               "coalition": sample(coalition, dtype = self._FLOAT),
               "proposal": proposal,
               "response": response,
               "time": self._INT(time),
               "proposer": proposer,
               "agent": proposer,
               "role": self._INT(role)}

        self.obs = obs

        return self.obs, {}

    def step(self, action: np.ndarray):

        self.obs["coalition"] = sample(action, dtype = self._FLOAT)
        proposal = proposal_heuristic(self.n_vehicles, action)
        self.obs["proposal"] = proposal
        response = response_heuristic(action, proposal, dtype = self._BIN)
        self.obs["response"] = response
        reward = self.reward()
        proposer = select(self.n_vehicles, dtype = self._INT)
        self.obs["proposer"] = proposer
        self.obs["agent"] = proposer
        self.obs["time"] -= 1
        self.obs["time"] = max(self.obs["time"], 0)
        terminated = False
        truncated = False
        info = {}

        if np.sum(self.obs["coalition"]) < 2:
            terminated = True

        return self.obs, reward, terminated, truncated, info

    def gain(self, coalition: np.ndarray):
        mask = np.all(coalition == self.coalitions, axis = 1)
        if not np.any(mask):
            return 0.0
        else:
            return self.gains[mask].astype(self._FLOAT).item()

    def reward(self, previous: np.ndarray | None = None):

        coalition = self.obs["coalition"]
        value = self.gain(coalition)
        i = self.obs["proposer"]
        n = 1 if np.sum(coalition) == 0 else np.sum(coalition)

        return (1 / n) * coalition[i] * value

    def terminated(self):

        coalition = self.obs["coalition"]
        response = self.obs["response"]
        _terminated = np.all(coalition == (response * coalition))

        return bool(_terminated)

"""
if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env
    from gymnasium.envs.registration import registry, register

    name = "Selection-v0"
    if name not in registry:
        register(id = name,
                 entry_point = Selection,
                 max_episode_steps = 10)

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