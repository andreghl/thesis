from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from bargain import generate_observation
from bargain.env.dummies import *
from typing import Any

import gymnasium as gym
import numpy as np

def get_actions_proposer(action: np.ndarray,
                         n: int,
                         dtype: type = np.float32):
    """Return the coalition selected and the proposal made by the proposer."""
    m = 2 * n
    coalition = action[:n].astype(dtype)
    proposal = action[n:m].astype(dtype)

    return coalition, proposal

class Bargain(gym.Env):

    _INT = np.int64
    _BIN = np.int8
    _BOOL = np.bool
    _FLOAT = np.float32

    def __init__(self,
                 n_vehicles: int,
                 n_customers: int,
                 max_time = 10):

        self.action_dim = 2 * n_vehicles + 1
        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.max_time = max_time
        self.max_steps = n_vehicles + 2
        self.timestep = 0

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
        role = 0

        obs = {"instance": flatten(self.instance[:, 1:], dtype = self._FLOAT),
               "coalition": coalition,
               "proposal": proposal,
               "response": response,
               "time": self.observation_space["time"].sample(),
               "proposer": proposer,
               "agent": proposer,
               "role": self._INT(role)}

        self.obs = obs
        return self.obs, {}

    def step(self, action: np.ndarray):

        role: int = self.obs["role"]
        agent = self.obs["agent"]
        done = self.done()
        _mask = mask(action, role, self.n_vehicles)
        reward = 0.0
        terminated = False

        if role == 0:
            coalition, proposal = get_actions_proposer(action,
                                                       self.n_vehicles,
                                                       self._FLOAT)

            self.obs["coalition"] = coalition.astype(self._BIN)
            self.obs["proposal"] = proposal
            self.obs["agent"] = self.order.pop()
            self.obs["role"] = 1
            self.obs["response"][agent] = self._BIN(1.0)

        if role == 1:
            coalition = self.obs["coalition"]
            self.obs["response"][agent] = action[_mask].astype(self._BIN)[0] * self._BIN(coalition[agent])

            if not self.order:
                self.obs["role"] = 2
                self.obs["agent"] = 0
            else:
                self.obs["agent"] = self.order.pop()

        if role == 2:
            reward = self.reward()
            terminated = self.terminated()
            self.obs["agent"] = (self.obs["agent"] + 1) % self.n_vehicles

            if self.obs["agent"] == 0:
                order = np.random.choice(np.arange(self.n_vehicles),
                                         size = self.n_vehicles,
                                         replace = False)
                self.order = list(order)
                proposer = self.order.pop()
                self.obs["proposer"] = proposer
                self.obs["agent"] = proposer
                self.obs["time"] -= 1
                self.obs["role"] = self._INT(0)

        truncated = self.truncated()
        return self.obs, float(reward * done), terminated, truncated, {}


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

        reward = coalition[i] * proposal[i] * response[i] * value
        # print("REWARD", coalition[i], proposal[i], response[i], value, i)

        return reward

    def terminated(self):

        coalition = self.obs["coalition"].copy()
        response = self.obs["response"].copy()
        agent = self.obs["agent"]

        last_agent = agent == self.n_vehicles - 1
        _terminated = np.all(coalition == (response * coalition))
        _feasible = np.sum(coalition) > 1

        return bool(_terminated and last_agent and _feasible)

    def done(self):

        coalition = self.obs["coalition"].copy()
        response = self.obs["response"].copy()
        role = self.obs["role"]

        _terminated = np.all(coalition == (response * coalition))
        _role = role == 2
        _feasible = np.sum(coalition) > 1

        # print("DONE", _terminated, _role, _feasible)

        return bool(_terminated and _role and _feasible)

    def truncated(self):
        time = self.obs["time"]
        if time < 0:
            self.obs["time"] = 0
        return bool(time < 0)


"""
if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env
    from gymnasium.envs.registration import registry, register
    import gymnasium as gym

    name = "Bargain-v0"
    if name not in registry:
        register(id = name,
                 entry_point = Bargain,
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