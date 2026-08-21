from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from bargain import generate_observation
from bargain.envs.dummies import *
from typing import Any

import gymnasium as gym
import numpy as np

class Bargain(gym.Env):

    _INT = np.int64
    _BIN = np.int8
    _BOOL = np.bool
    _FLOAT = np.float32

    def __init__(self,
                 n_vehicles: int,
                 n_customers: int,
                 max_time = 10,
                 active: str = "selection",
                 agents_path: str = "data/agents"):

        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.max_time = max_time
        self.path = agents_path
        self.active = active
        self.max_steps = n_vehicles + 2
        self.timestep = 0
        self.roles = {"selection": 0, "proposal": 1, "response": 2}

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


        role = self.roles[self.active]

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

        if self.active == "selection":
            a = 1

        return 0






