import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def cost(routes : list, distance : np.ndarray):
    """Return the total distance of a single route."""

    _cost = 0
    for i, j in zip(routes, routes[1:]):
        _cost += distance[i, j]

    return _cost

def distances(instance : np.ndarray):
    """Return the distance between every (u, v) pair in the graph.

    Args:
        instance: a (n x 5) vehicle routing instance matrix containing
        the id, position, an indicator of whether a node is a depot,
        and the vehicle assigned to each node.

    Returns:
        A (n x n) matrix containing the distance between any pair of
        nodes (i, j) indexed by their id.
    """
    n = instance.shape[0]
    distance = np.zeros((n, n), dtype = np.float64)

    # TODO: Find a way to vectorize the loop (?efficiency)
    for i in range(n):
        for j in range(n):
            distance[i, j] = np.linalg.norm(instance[i, 1:3] - instance[j, 1:3])

    return distance

def gain(routes : dict[int, list], initial_routes : dict[int, list], distance : np.ndarray):
    """Compute the collaboration gain obtained by a coalition.

    Args:
        routes: dictionary of lists containing the ids of the nodes in the routes
        of each vehicle indexed by the vehicle id after a coalition has formed and
        shared nodes.
        initial_routes: dictionary of lists containing the ids of the nodes in the
        initial routes of each vehicle.
        distance: a numpy matrix containing the distance between each pair of nodes
        in the vehicle routing instance matrix.

    Returns:
        A float representing the difference between the total distance of the initial
        routes and the current collaboration routes.
    """
    _gain = 0
    depots = set(routes.keys()).union(initial_routes.keys())
    for depot in depots:
        _gain += cost(initial_routes[depot], distance) - cost(routes[depot], distance)

    return _gain

def plot(instance : np.ndarray, routes : dict, coalition : list | set, ax : matplotlib.axes._axes.Axes):
    """Plot of the routes obtained using the Clarke-Wright savings algorithm."""

    node_colors = ['grey', 'red']
    depot_col = 3

    for i, color in enumerate(node_colors):
        nodes = instance[np.isin(instance[:, depot_col], i), 1:3]
        ax.scatter(*nodes.T, c = color)
    ax.set_title(label = f"Instance with coalition: {coalition}")

    depot_value = 1
    n_depots = len(instance[np.isin(instance[:, depot_col], depot_value)])
    cmap = plt.get_cmap('tab20')
    route_colors = [cmap(i) for i in np.linspace(0,1, n_depots)]

    for (depot, route), color in zip(routes.items(), route_colors):
        indexes = route.astype(int)
        ax.plot(*instance[indexes, 1:3].T,
                 label = f"depot {depot}",
                 c = color)

    ax.legend()
    ax.grid(True)