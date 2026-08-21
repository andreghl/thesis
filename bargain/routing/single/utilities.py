from .heuristics import clarke_wright
from .instance import map_col, get_vehicles

import numpy as np
import itertools


def cost(route : list, distance : np.ndarray):
    """Return the cost of a single route."""
    _cost = 0
    for i in range(len(route) - 1):
        _cost += distance[route[i], route[i + 1]]

    return _cost

def yield_coalitions(players : list):
    """Generate the 2^N coalitions for a list of N players."""
    n = len(players)
    for size in range(n + 1):
        for coalition in itertools.combinations(range(n), size):
            yield list(coalition)

def gains(instance : np.ndarray, distance : np.ndarray):
    """Compute the collaboration gain obtained by a coalition.

    Args:
        instance: a (n x 5) VRP instance matrix.
        distance: a (n x n) distance matrix.

    Returns:

    """
    vehicles = get_vehicles(instance)
    gain = {}; routes = {}

    for i, coalition in enumerate(yield_coalitions(vehicles)):
        routes[frozenset(coalition)] = clarke_wright(instance,
                                                     distance,
                                                     coalition)

        pre_revenue = [len(routes[frozenset([v])][v]) - 2
                       for v in coalition]
        pre_cost = [cost(routes[frozenset([v])][v], distance)
               for v in coalition]
        pre_profit = sum(pre_revenue) - sum(pre_cost)

        post_revenue = [len(routes[frozenset(coalition)][v]) - 2
                        for v in coalition]
        post_cost = [cost(routes[frozenset(coalition)][v], distance)
                for v in coalition]
        post_profit = sum(post_revenue) - sum(post_cost)

        gain[frozenset(coalition)] = sum(pre_cost) - sum(post_cost)

    return gain, routes
