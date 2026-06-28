import numpy as np


def get_players(instance : np.ndarray):
    """Return the ID of players (depots) in an instance matrix."""

    id_col = 0
    depot_col = 3
    depot_value = 1

    mask = np.isin(instance[:, depot_col], depot_value)
    depots = instance[mask, id_col]

    return depots