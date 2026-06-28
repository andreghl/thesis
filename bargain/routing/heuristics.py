import numpy as np
import itertools


def assign_customers(instance : np.ndarray,
                     distance : np.ndarray,
                     coalition : list):
    """Return an assignment of vehicles/depots to nodes in the graph for heuristic routing.

    Args:
        instance: a (n x 5) vehicle routing instance matrix.
        distance: a (n x n) matrix containing the distance between any two node in the instance matrix.
        coalition: a list of players (integers) involved in the coalition

    Returns:
        A copy of the instance matrix with an assigment of minimal cost in the last column.
    """

    id_col = 0
    pos_col = (1, 2)
    depot_col = 3
    vehicle_col = 4

    instance = instance.copy()
    depots_mask = np.isin(instance[:, depot_col], test_elements = 1)
    depots = instance[depots_mask, id_col].astype(int)

    weights = distance[:, :len(depots)]
    weights_sum = weights.sum(axis = 1)[:, None]
    weights = (weights / weights_sum).round(decimals = 4)
    assignment = np.argmin(weights, axis = 1)

    in_coalition_assigned = np.isin(assignment, coalition)
    in_coalition_initial = np.isin(instance[:, vehicle_col], coalition)

    mask = in_coalition_assigned & in_coalition_initial
    instance[mask, vehicle_col] = assignment[mask]

    return instance


def clarke_wright(instance : np.ndarray,
                  distance : np.ndarray):
    """Return the route of each vehicle in the vehicle routing instance.

    Args:
        instance: a (n x 5) vehicle routing instance matrix.
        distance: a (n x n) matrix containing the distance between any two node in the instance matrix.

    Returns:
        A dictionary containing a list for every vehicle representing the route taken by the vehicle.
    """

    id_col = 0
    pos_col = (1, 2)
    depot_col = 3
    vehicle_col = 4

    depots_mask = np.isin(instance[:, depot_col], test_elements = 1)
    depots = instance[depots_mask, id_col].astype(int)
    routes = {depot: None for depot in depots}

    for depot in depots:
        savings = []

        # identify the customers assigned to the depot under consideration
        mask = np.isin(instance[:, depot_col], test_elements = 0)
        mask = mask * np.isin(instance[:, vehicle_col], depot)
        customers = instance[mask, id_col].astype(int)

        route = {k: [v] for k, v in enumerate(customers)}
        for i, j in itertools.permutations(customers, r = 2):
            if i != j:
                saving = distance[depot, i] + distance[depot, j] - distance[i, j]
                savings.append((i, j, saving))

        savings.sort(key = lambda x : x[-1], reverse = True)

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
