from src.utils.heuristics import assign_customers

import numpy as np
import itertools


def clarke_wright(dm : np.ndarray,
                  dist : np.ndarray,
                  depots : list,
                  coalition : list | None = None):
    """
    This method
    """

    routes = {depot: None for depot in depots}
    _dm = assign_customers(dm, dist, depots, coalition)

    for depot in depots:
        savings = []

        # identify the customers assigned to the depot under consideration
        mask = np.isin(_dm[:, 3], 0) * np.isin(_dm[:, 4], depot)
        customers = _dm[mask, 0].astype(int)

        route = {k: [v] for k, v in enumerate(customers)}
        for i, j in itertools.permutations(customers, 2):
            if i != j:
                saving = dist[depot, i] + dist[depot, j] - dist[i, j]
                savings.append((i, j, saving))

        savings.sort(key = lambda x: x[-1], reverse = True)

        for i, j, s in savings:
            x = y = None
            for k, v in route.items():
                if v[-1] == i:
                    x = k

                if v[0] == j:
                    y = k

            if x is not None and y is not None and x != y:
                new_route = route[x] + route[y]
                route[x] = new_route
                del route[y]

        routes[depot] = np.array([depot] + list(*route.values()) + [depot])

    return routes