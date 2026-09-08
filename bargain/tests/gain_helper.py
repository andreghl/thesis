from bargain import generate_observation
from bargain.networks import GainNN, get_device, transform

import numpy as np
import torch


def test_gain_net(path : str, seed : int | None = None):
    """Return the predicted and actual values of a randomly generated instance.

    Args:
        path: a string indicating the location of a .pth file.
        seed: an integer controlling the torch and numpy randomness.

    Returns:
        A vector with the predicted values and another with the actual values
        of the characteristic function.
    """

    device = get_device()

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = GainNN().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    instance, _, coalition, char_function, _, _ = generate_observation(
        n_vehicles = 3, n_customers = 9, radius = [0.3, 0.4, 0.6]
    )

    # transform the features to fit the neural network
    instance = instance.reshape(1, 10, 5)
    coalition = np.array(coalition).reshape(1, 4, 3)
    char_function = np.array(char_function).reshape(1, 4)

    instance, coalition, char_function = transform(
        instance, coalition, char_function, device
    )

    with torch.no_grad():
        values = model(instance.float(), coalition.float())

    return values.t(), char_function.t()

def tests_gain_net(path: str, n_tests: int, seed: int | None = None):
    """Return the predicted and actual values of a set of randomly generated instances.

    Args:
        path: a string indicating the location of a .pth file.
        n_tests: an integer defining the number of inputs provided to the network.
        seed: an integer controlling the torch and numpy randomness.

    Returns:
        A vector with the predicted values and another with the actual values
        of the characteristic function.
    """

    values = np.zeros(shape = (n_tests, 1, 4), dtype = np.float64)
    gains = np.zeros(shape = (n_tests, 1, 4), dtype = np.float64)

    for n in range(n_tests):
        _values, _gains = test_gain_net(path, seed)
        values[n] = _values
        gains[n] = _gains

    return values.reshape(n_tests * 4, ), gains.reshape(n_tests * 4, )

