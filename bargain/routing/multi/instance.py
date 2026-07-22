import numpy as np


def generate_depots(n_depots : int):
    """Return an array containing the location of up to three depots on a 2D plane."""

    depots_loc = np.array([(-0.2, 0.173), (0.2, 0.173), (0, -0.173)])
    k = len(depots_loc)
    n_depots = min(n_depots, len(depots_loc))
    indexes = np.random.choice(range(k), size = n_depots, replace = False)

    return depots_loc[indexes]

def generate_customers(n_customers : int, radius : float, decimals : int = 3):
    """Return an array of randomly located customer nodes on a 2D plane."""
    return np.random.uniform(-radius, radius, (n_customers, 2)).round(decimals)

def generate_instance(n_depots : int = 3,
                      n_customers : int = 9,
                      radius : float = 0.3):
    """Return a matrix of depots and customers on a 2D plane.

    Args:
        n_depots: an integer defining the number of depots in the instance (max = 3).
        n_customers: an integer determining the number of customers in the instance.
        radius: a float impacting the degree of overlap between the customers
        and the degree of collaboration opportunities.

    Returns:
        A (n x 5) matrix containing 5 columns, namely the id of the node (0),
        the x position of the node (1), the y position of the node (2),
        a binary indicator of whether the node is a depot (3), and an assignment
        of vehicle id to the nodes (4).
    """

    # mapping of information to matrix column
    _map = {'id': 0, 'pos': (1, 2), 'depot': 3, 'vehicle': 4}
    depots = generate_depots(n_depots)
    customers = generate_customers(n_customers, radius)

    n_depots = len(depots)
    n_customers = len(customers)

    n_nodes = n_depots + n_customers
    instance = np.zeros(shape = (n_nodes, 5))
    instance[:, _map['id']] = np.arange(n_nodes)
    instance[:, _map['pos']] = np.concatenate((depots, customers))
    instance[:, _map['depot']] = np.concatenate((np.ones(n_depots),
                                            np.zeros(n_customers)))
    instance[:, _map['vehicle']] = np.tile(np.arange(n_depots), n_customers)[:n_nodes]

    return instance