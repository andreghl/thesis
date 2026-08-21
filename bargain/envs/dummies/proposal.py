from bargain.envs.dummies import coalition_heuristic, proposal_heuristic, response_heuristic
from bargain.envs.dummies import Selection, select, flatten, softmax
from gymnasium.spaces import Box
from typing import Any
import numpy as np


class Proposal(Selection):

    def __init__(self,
                 n_vehicles: int,
                 n_customers: int,
                 max_time: int = 10):
        super().__init__(n_vehicles, n_customers, max_time)

        self.action_space = Box(low = 0,
                                high = 1,
                                shape = (self.n_vehicles, ),
                                dtype = self._FLOAT)

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed = seed, options = options)

        proposer = select(self.n_vehicles, dtype = self._INT)
        coalition = coalition_heuristic(self.coalitions, self.gains, dtype = self._BIN)
        proposal = proposal_heuristic(self.n_vehicles, coalition, dtype = self._FLOAT)
        response = response_heuristic(coalition, proposal, dtype = self._BIN)
        time = select(self.max_time - 1, dtype = self._INT)
        role = 1

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

        coalition = coalition_heuristic(self.coalitions,
                                        self.gains,
                                        dtype = self._BIN)
        self.obs["coalition"] = coalition
        self.obs["proposal"] = softmax(action)
        response = response_heuristic(coalition, action, dtype = self._BIN)
        self.obs["response"] = response
        reward = self.reward()
        proposer = select(self.n_vehicles, dtype = self._INT)
        self.obs["proposer"] = proposer
        self.obs["agent"] = proposer
        self.obs["time"] = max(self.obs["time"] - 1, 0)
        terminated = self.terminated()
        truncated = False
        info = {}

        return self.obs, reward, terminated, truncated, info

    def reward(self):
        one = 1
        time = self.obs["time"]
        if time == 0:
            one = 0

        coalition = self.obs["coalition"]
        proposal = self.obs["proposal"]
        response = self.obs["response"]
        i = self.obs["proposer"]
        reward = coalition[i] * proposal[i] * response[i]
        reward += np.sum(coalition * proposal * response)

        return float(reward * one)

"""
if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env
    from gymnasium.envs.registration import registry, register
    import gymnasium as gym

    name = "Proposal-v0"
    if name not in registry:
        register(id = name,
                 entry_point = Proposal,
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