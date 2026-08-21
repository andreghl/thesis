from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from .utils import Observation, is_truncated
from typing import Any

import gymnasium as gym
import numpy as np


class Proposal(gym.Env):

    def __init__(self,
                 obs: Observation):

        self.obs = obs
        self.obs.active = "proposal"
        self.action_space = Box(
            low = 0,
            high = 1,
            shape = (obs.n_vehicles, ),
            dtype = obs.float
        )
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
            "role": Discrete(n = 3, dtype = obs.int)}
        )

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed = seed, options = options)
        obs = self.obs.reset()

        return obs, {}

    def step(self, action: np.ndarray):
        coalition = self.obs["coalition"]
        obs = self.obs.step()
        proposal = self.obs.check_nan(action, coalition)
        self.obs["proposal"] = proposal.copy()
        obs = self.obs.step()
        # print("step", proposal, action, self.obs["proposal"])
        reward, terminated = self.reward(proposal)
        truncated = is_truncated(self.obs["time"])
        # print("step2", proposal, action, self.obs["proposal"])

        return obs, reward, terminated, truncated, {}

    def reward(self, proposal: np.ndarray):
        # print("reward", proposal)

        agent = self.obs["agent"]
        coalition = self.obs["coalition"]

        while self.obs["role"] != 1:

            while self.obs["role"] == 2:
                self.obs.respond(self.obs)
                self.obs.step()

            while self.obs["role"] == 0:
                self.obs.select(self.obs)
                self.obs.step()

        response = self.obs["response"]
        value = self.obs.value(coalition)
        done = np.all(coalition == response)
        reward = coalition[agent] * proposal[agent] * response[agent] * value

        self.obs.internal_reset()
        return float(reward * (int(done) + 0.2)), bool(done)


if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env
    from gymnasium.envs.registration import registry

    if "Proposal-v0" not in registry:
        gym.register(
            id = "Proposal-v0",
            entry_point = Proposal,
            max_episode_steps = 10
        )

    vehicles = 3
    customers = 9
    maxtime = 10
    observation = Observation(vehicles, customers, maxtime)

    # Discrete action space environment
    env = gym.make(id = "Proposal-v0",
                   obs = observation)

    try:
        check_env(env,
                  warn = False,
                  skip_render_check = True); print("Passed!")

    except Exception as e:
        print(f"Failed!\n")
        print(e)