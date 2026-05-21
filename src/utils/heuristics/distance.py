import numpy as np

def distances(dm : np.ndarray):
    """
    The 'distances' method takes a (|V| x 5) VRP matrix where V is the set of nodes and the columns are defined as
        0: id of node.
        1: x coordinate of the node in a Cartesian plane.
        2: y coordinate of the node in a Cartesian plane.
        3: binary indicator of whether the node is a depot.
        4: Assignment of nodes to depots/vehicles (using depots' IDs)
    """
    size = dm.shape[0]
    dist = np.zeros((size, size), dtype = np.float64)
    for i in range(size):
        for j in range(size):
            dist[i, j] = np.linalg.norm(dm[i, 1:3] - dm[j, 1:3])

    return dist