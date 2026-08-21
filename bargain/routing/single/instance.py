import numpy as np


def map_col(name : str):
    """Return the column of a named variable."""
    _map = {'id': 0, 'pos': (1, 2), 'depot': 3, 'vehicle': 4}

    if name not in _map:
        raise ValueError(f"Argument {name} not found in {tuple(_map.keys())}")

    return _map[name]

def get_depot(instance : np.ndarray):
    """Return the row of the depot."""
    return instance[np.isin(instance[:, map_col('depot')], test_elements = 1)]

def get_vehicles(instance : np.ndarray, return_counts : bool = False):
    """Return the ids of the vehicles in the instance."""
    mask = ~ np.isnan(instance[:, map_col('vehicle')])
    return np.unique(instance[mask, map_col('vehicle')], return_counts = return_counts)

def generate_depot():
    """Return the location of the depot."""
    return np.zeros(shape = (1, 2))

def generate_customers(n_customers : int, radius : float, decimals : int = 3):
    """Return an array of randomly located customer nodes on a 2D plane."""
    return np.random.uniform(-radius, radius, (n_customers, 2)).round(decimals)

def generate_instance(n_vehicles : int = 3,
                      n_customers : int = 9,
                      radius : float = 0.3):
    """"""

    n_depot = 1; n_col = 5
    n_nodes = n_customers + n_depot
    depot = generate_depot()
    customers = generate_customers(n_customers, radius)

    instance = np.zeros(shape = (n_nodes, n_col))
    instance[:, map_col('id')] = np.arange(n_nodes)
    instance[:, map_col('pos')] = np.concatenate((depot, customers))
    instance[:, map_col('depot')] = np.concatenate((np.ones(n_depot),
                                                 np.zeros(n_customers)))
    instance[1:n_nodes, map_col('vehicle')] = np.tile(np.arange(n_vehicles), n_customers)[:n_customers]
    instance[0, map_col('vehicle')] = None

    return instance.astype(np.float64)