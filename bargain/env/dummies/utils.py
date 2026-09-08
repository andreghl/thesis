from stable_baselines3 import PPO
import numpy as np
import os


def flatten(instance: np.ndarray, dtype: type):
    """Flattens the instance matrix for neural network manipulation."""
    assert len(instance.shape) == 2, f"instance does not contain two dims {instance.shape}."
    rows, cols = instance.shape
    return instance.reshape(1, rows * cols).astype(dtype)

def one_hot(index: np.ndarray | int, size: int, dtype: type = np.int8):
    x = np.zeros(size, dtype = dtype)
    x[index] = 1
    return x.astype(dtype)

def softmax(x: np.ndarray, axis: int = -1):
    x = np.exp(x)
    x_sum = np.sum(x, axis = axis, keepdims = True)
    return x / x_sum

def coalition_heuristic(coalitions: np.ndarray,
                        gain: np.ndarray,
                        dtype: type = np.int8):
    """Randomly select a suitable coalition proportionally to its gain."""
    k, _ = coalitions.shape
    p = np.maximum(gain, 0)
    p = softmax(p)
    x = np.arange(k)
    coalition = coalitions[np.random.choice(x, p = p)].astype(dtype)
    return coalition

def proposal_heuristic(n: int,
                       coalition: np.ndarray,
                       previous_proposal: np.ndarray | None = None,
                       previous_response: np.ndarray | None = None,
                       dtype: type = np.float32):
    proposal = np.random.uniform(size = n)
    proposal = proposal * coalition

    if previous_proposal is not None and previous_response is not None:
        x = previous_proposal * np.abs(previous_response - 1)
        noise = np.random.normal(loc = 0, scale = x, size = x.shape)
        proposal = previous_proposal + np.abs(noise)
    return softmax(proposal).astype(dtype)

def response_heuristic(coalition: np.ndarray, proposal: np.ndarray, dtype: type = np.int8):
    n = len(coalition)
    k = n if np.sum(coalition) == 0 else np.sum(coalition)
    response = np.ones(n) / k
    response = response * coalition
    noise = np.random.normal(loc = 0, scale = response, size = response.shape)
    response + noise
    return (response - 1e-5 <= proposal).astype(dtype)

def mask(action: np.ndarray, role: int, n_vehicles: int):
    _mask = np.ones(action.shape[-1])
    if role == 0:
        _mask[-1] = False
    elif role == 1:
        _mask[:-1] = False
    elif role == 2:
        _mask[:] = False
    else:
        print(f"> role {role} not in {0, 1, 2}")

    return _mask.astype(bool)

def get_actions_proposer(action: np.ndarray,
                         n: int,
                         dtype: tuple[type, type] = (np.int8, np.float32)):
    """Return the coalition selected and the proposal made by the proposer."""
    m = 2 * n
    c_type, p_type = dtype
    coalition = action[:n].astype(c_type)
    proposal = action[n:m].astype(p_type)

    return coalition, proposal

def response(state: dict, order: list[int], n_vehicles: int, agent_path: str | None):

    _response = response_heuristic(state["coalition"], state["proposal"])
    if agent_path is not None and os.path.exists(agent_path + ".zip"):
        model = PPO.load(agent_path, tensorboard_log = None)

        while order:
            state = state.copy()
            state["role"] = 1
            state["agent"] = order.pop()
            action, _ = model.predict(state)
            action = action[mask(action, state["role"], n_vehicles)][0]
            agent = state["agent"]
            _response[agent] = action

    return _response

def propose(state: dict | None,
            n_vehicles: int,
            agent_path: str | None,
            coalitions: np.ndarray,
            gains: np.ndarray,
            previous_proposal: np.ndarray | None = None,
            previous_response: np.ndarray | None = None):

    coalition = coalition_heuristic(coalitions, gains)
    proposal = proposal_heuristic(n_vehicles, coalition, previous_proposal, previous_response)

    if agent_path is not None and os.path.exists(agent_path + ".zip"):
        model = PPO.load(agent_path, tensorboard_log = None)

        if state is not None:
            state = state.copy()
            state['role'] = 0
            action, _ = model.predict(state)
            coalition, proposal = get_actions_proposer(action, n_vehicles)

    return coalition, proposal




