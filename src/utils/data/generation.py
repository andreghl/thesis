from src.utils import instance, cost, gain, plot
from src.utils.heuristics import clarke_wright, assign_customers, distances
from src.utils.solutions import nucleolus, shapley

import numpy as np
import itertools


def encode_coalition(coalition : list, n_depots : int):
    encoding = np.zeros(n_depots, dtype = np.int32)
    encoding[coalition] = 1
    return encoding

def generate(n_depots : int, n_customers : int, plots : bool = False):
    _, depots, _, nodes = instance((n_customers, n_depots))
    players = list(range(n_depots))
    n_nodes = n_depots + n_customers

    # id_node : id of nodes.
    id_node = np.arange(n_nodes).reshape(n_nodes, 1)
    # pos : x, y position of nodes.
    pos = np.array(depots + nodes)
    # depot : binary indicator of depot.
    depot = np.concatenate([np.ones((n_depots, 1)),
                            np.zeros((n_customers, 1))]
                           )
    # vehicle : assignment of nodes to depot/vehicle.
    vehicle = np.concatenate([np.arange(n_depots).reshape(n_depots, 1),
                              np.random.choice(range(n_depots), size = (n_customers, 1))]
                             )

    # create distance matrix of Mak et al. (2023)
    dm = np.concatenate([id_node, pos, depot, vehicle], axis = 1)
    dist = distances(dm)

    # create variable for characteristic function
    v = {}
    coalitions = []
    index = 0

    # compute the pre-collaboration routes
    pre_routes = clarke_wright(dm, dist, players)
    if plots:
        plot(dm, pre_routes, colab = ())

    # store assignment of vehicles to nodes
    n_coalitions = 2 ** n_depots
    assignments = np.zeros((n_nodes, n_coalitions), dtype = np.int32)

    # compute the collaboration routes (and gain) for each coalition
    for size in range(n_depots + 1):
        for coalition in itertools.combinations(range(n_depots), size):

            coalitions.append(encode_coalition(list(coalition), n_depots))
            _dm = assign_customers(dm, dist, players, list(coalition))
            assignments[:, index] = _dm[:, -1]
            index += 1

            # set singleton coalitions to 0
            if coalition in [(), (0, ), (1, ), (2, )]:
                v[frozenset(coalition)] = 0
                continue

            routes = clarke_wright(dm, dist, players, list(coalition))

            pre_cost = 0
            for depot in coalition:
                pre_cost += cost(pre_routes[depot], dist)
            v[frozenset(coalition)] = gain(routes, pre_routes, dist, coalition)
            if plots:
                plot(dm, routes, coalition)


    try:
        _nucleolus = nucleolus(players, v).values()
    except RuntimeError:
        _nucleolus = np.zeros(n_depots)

    coalitions = np.array(coalitions)
    _shapley = np.array(list(shapley(players, v).values()))
    _nucleolus = np.array(list(_nucleolus))
    v = np.array(list(v.values()))

    return dm, assignments, coalitions, v, _shapley, _nucleolus

