from .heuristics import clarke_wright
from .instance import get_vehicles

import matplotlib.pyplot as plt
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

def plot(instance : np.ndarray, routes : dict, coalition : list | set, ax : matplotlib.axes._axes.Axes):
    """Plot of the routes obtained using the Clarke-Wright savings algorithm."""

    node_colors = ['grey', 'red']
    depot_col = 3

    for i, color in enumerate(node_colors):
        nodes = instance[np.isin(instance[:, depot_col], i), 1:3]
        ax.scatter(*nodes.T, c = color)
    ax.set_title(label = f"Instance with coalition: {coalition}")

    depot_value = 1
    vehicle_col = -1
    mask = instance[:, depot_col] != depot_value
    n_vehicles = len(np.unique(instance[mask, vehicle_col]))
    cmap = plt.get_cmap('tab20')
    route_colors = [cmap(i) for i in np.linspace(0,1, n_vehicles)]

    _routes = {vehicle: [] for vehicle in range(n_vehicles)}

    for i in range(n_vehicles):

        if i in coalition:
            _routes[i] = routes[frozenset(coalition)][i]

        else:
            _routes[i] = routes[frozenset([i])][i]

    for (vehicle, route), color in zip(_routes.items(), route_colors):
        indexes = list(map(int, route))
        ax.plot(*instance[indexes, 1:3].T,
                label = f"vehicle {vehicle}",
                c = color)

    ax.legend()
