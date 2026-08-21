from stable_baselines3 import PPO
import numpy as np
import os


def sample(p: np.ndarray, dtype: type):
    return np.random.binomial(n = 1, p = p).astype(dtype)

def flatten(instance: np.ndarray, dtype: type):
    """Flattens the instance matrix for neural network manipulation."""
    assert len(instance.shape) == 2, f"instance does not contain two dims {instance.shape}."
    rows, cols = instance.shape
    return instance.reshape(1, rows * cols).astype(dtype)

def check_path(path: str, model: str):
    """Check whether a PPO zip file exists at a given location."""
    x = None
    if os.path.exists(path + model + ".zip"):
        x = PPO.load(path + model); print(f"model '{path + model}' loaded.")
    return x

def one_hot(index: np.ndarray | int, size: int, dtype: type = np.int64):
    x = np.zeros(size, dtype = dtype)
    x[index] = 1
    return x.astype(dtype)

def select(n: int, dtype: type):
    return np.random.choice(np.arange(n, dtype = dtype))

def softmax(x: np.ndarray, axis: int = -1):
    x = np.exp(x)
    x_sum = np.sum(x, axis = axis, keepdims = True)
    return x / x_sum

def coalition_heuristic(coalitions: np.ndarray,
                        gain: np.ndarray | None = None,
                        dtype: type = np.int64):
    if gain is None:
        k, _ = coalitions.shape
        x = np.arange(k)
        coalition = coalitions[np.random.choice(x)].astype(dtype)
    else:
        k, _ = coalitions.shape
        p = np.maximum(gain, 0)
        p = softmax(p)
        x = np.arange(k)
        coalition = coalitions[np.random.choice(x, p = p)].astype(dtype)

    return coalition

def proposal_heuristic(n: int,
                       coalition: np.ndarray,
                       dtype: type = np.float32,
                       previous_proposal: np.ndarray | None = None,
                       previous_response: np.ndarray | None = None):
    proposal = np.random.uniform(size = n)
    proposal = proposal * coalition

    if previous_proposal is not None and previous_response is not None:
        x = previous_proposal * np.abs(previous_response - 1)
        noise = np.random.normal(loc = 0, scale = x, size = x.shape)
        proposal = previous_proposal + noise
    return softmax(proposal).astype(dtype)

def response_heuristic(coalition: np.ndarray, proposal: np.ndarray, dtype: type = np.int64):
    n = len(coalition)
    k = n if np.sum(coalition) == 0 else np.sum(coalition)
    response = np.ones(n) / k
    response = response * coalition
    return (response <= proposal).astype(dtype)

def mask(action: np.ndarray, role: int, n_vehicles: int):
    _mask = np.ones(action.shape)
    if role == 0:
        _mask[n_vehicles:] = 0
    elif role == 1:
        _mask[:n_vehicles] = 0
        _mask[-1] = 0
    elif role == 2:
        _mask[:-1] = 0
    else:
        print(f"> role {role} not in {0, 1, 2}")

    return action * _mask

def selection(obs: dict,
              dtype: type,
              path: str = "data/agents"):

    if os.path.exists(path + "selection.zip"):
        a = 1
    else:
        a = 2
    return a