from bargain import generate_observation
from stable_baselines3 import PPO
from collections import deque

import numpy as np
import os

def sample(probs: np.ndarray):
    return np.random.binomial(n = 1, p = probs, size = probs.shape).astype(np.float32)

def flatten(instance: np.ndarray):
    """Flattens the instance matrix for neural network manipulation."""
    assert len(instance.shape) == 2, f"instance does not contain two dims {instance.shape}."
    rows, cols = instance.shape
    return instance.reshape(1, rows * cols)

def check_path(path: str, model: str):
    """Check whether a PPO zip file exists at a given location."""
    x = None
    if os.path.exists(path + model + ".zip"):
        x = PPO.load(path + model); print(f"model '{path + model}' loaded.")
    return x

def one_hot(index: np.ndarray | int, size: int, dtype = np.int64):
    x = np.zeros(size, dtype = dtype)
    x[index] = 1
    return x

def need_new_proposer(timestep: int, n_vehicles):
    return timestep % (n_vehicles + 2) == 0

def is_truncated(time: int):
    return bool(time <= 0)

def next_role(timestep: int, n_vehicles: int):
    proposer_actions = 2
    total = n_vehicles + 2

    return min(timestep % total, proposer_actions)

def next_proposer(n_vehicles):
    return np.random.choice(
            np.arange(n_vehicles, dtype = np.int64)
        )

def next_agent(timestep: int, current: int, n_vehicles: int):
    if next_role(timestep, n_vehicles) == 2:
        return (current + 1) % n_vehicles
    else:
        return current

def next_time(time: int):
    return time - 1

class Observation(dict):

    _SIZE: int = 100000
    _INT = np.int64
    _FLOAT = np.float32
    _BOOL = np.bool

    def __init__(self,
                 n_vehicles: int,
                 n_customers: int,
                 max_time: int = 10,
                 model_path: str = "data/agents/",
                 threshold: int = 50000,
                 random_time: bool = False):
        super().__init__()

        self.n_vehicles = n_vehicles
        self.n_customers = n_customers
        self.max_time = max_time
        self.path = model_path
        self.threshold = threshold
        self.random = random_time
        self.timestep = 0

        # alternative PPO models to engage in 'self-play'
        self.coalition = check_path(model_path, model = "selection")
        self.proposal = check_path(model_path, model = "proposal")
        self.response = check_path(model_path, model = "response")
        self._buffer = deque(maxlen = self._SIZE)
        self.active = ""

        # utilities
        self.instance = None
        self.routes = None
        self.coalitions = None
        self.gains = None
        self.shapley = None
        self.nucleolus = None

        # formats
        self.float = self._FLOAT
        self.bool = self._BOOL
        self.int = self._INT

    def __copy__(self):
        return dict(self)

    def reset(self, seed: int | None = None):
        np.random.seed(seed)

        (self.instance,
         self.routes,
         self.coalitions,
         self.gains,
         self.shapley,
         self.nucleolus) = generate_observation(
            self.n_vehicles,
            self.n_customers,
            radius = 1.0
        )

        agent = np.random.choice(np.arange(self.n_vehicles, dtype = self._INT))
        proposer = np.random.choice(np.arange(self.n_vehicles, dtype = self._INT))

        # self.timestep = 0
        self["instance"] = flatten(self.instance[:, 1:]).astype(self._FLOAT)
        self["coalition"] = one_hot(proposer,
                                    size = self.n_vehicles,
                                    dtype = self._FLOAT)
        self["proposal"] = np.zeros(shape = (self.n_vehicles, ),
                                    dtype = self._FLOAT)
        self["response"] = one_hot(proposer,
                                   size = self.n_vehicles,
                                   dtype = self._FLOAT)
        self["time"] = self._INT(np.random.choice(np.arange(self.max_time))) if self.random else self._INT(
            self.max_time - 1)
        self["proposer"] = self._INT(proposer)
        self["agent"] = self._INT(agent)
        self["role"] = self._INT(0)

        return dict(self)

    def step(self):
        self.timestep += 1
        self["role"] = next_role(self.timestep,
                                 self.n_vehicles)
        self["agent"] = next_agent(self.timestep,
                                   self["agent"],
                                   self.n_vehicles)

        return dict(self)

    def internal_reset(self):

        proposer = need_new_proposer(self.timestep, self.n_vehicles)

        if proposer and not is_truncated(self["time"]):
            proposer = next_proposer(self.n_vehicles)
            self["time"] = next_time(self["time"])
            self["proposer"] = proposer
            self["agent"] = proposer

            if self.active in ["proposal", "response"]:
                self.select(self)
                self.step()

            if self.active == "response":
                self.propose(self)
                self.step()

        return dict(self)

    def check_nan(self,
                  proposal: np.ndarray,
                  coalition: np.ndarray,
                  eps: float = 1e-8):
        """Check if a proposal sums to 0."""
        _p = proposal.copy()
        if np.sum(coalition) == 0:
            return np.zeros(self.n_vehicles,
                            dtype = self._FLOAT)

        proposal = proposal * coalition

        if np.sum(proposal) > eps:
            proposal = proposal / np.sum(proposal)

        else:
            proposal = np.ones(self.n_vehicles,
                               dtype = self._FLOAT) * coalition
            proposal = proposal / np.sum(coalition)

        return proposal

    def value(self, coalition: np.ndarray):
        mask = np.all(coalition == self.coalitions, axis = 1)
        if not np.any(mask):
            return 0.0
        else:

            return self.gains[mask].astype(self._FLOAT).item()

    def select(self, state: dict):

        probs = np.abs(self.gains)
        n = len(self.gains)
        probs = np.ones(n) / n if np.sum(probs) == 0 else probs / np.sum(probs)
        index = np.random.choice(range(len(self.coalitions)), p = probs)
        coalition = np.array(self.coalitions[index])

        greedy_train = self.timestep < self.threshold
        if self.coalition is not None and not greedy_train:
            coalition, _ = self.coalition.predict(state)

        coalition = sample(coalition)
        state["coalition"] = coalition.copy()

        return state

    def propose(self, state: dict):

        coalition = self["coalition"]
        proposal = np.zeros(self.n_vehicles)

        greedy_train = self.timestep < self.threshold
        if self.proposal is not None and not greedy_train:
            proposal, _ = self.proposal.predict(state)

        proposal = self.check_nan(proposal, coalition)
        state["proposal"] = proposal.copy()

        return state

    def respond(self, state: dict):

        agent = state["agent"]
        coalition = state["coalition"]
        proposal = state["proposal"]
        response = np.ones(self.n_vehicles) * coalition
        response = proposal >= response
        response = response * one_hot(agent, self.n_vehicles)

        greedy_train = self.timestep < self.threshold
        if self.response is not None and not greedy_train:
            response, _ = self.response.predict(state)

        response = sample(response)
        response = np.maximum(state["response"], response)
        state["response"] = response.copy()

        return state
