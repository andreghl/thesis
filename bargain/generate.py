from .routing.single.heuristics import distances, assign_routes
from .routing.single.utilities import cost, gains
from .routing.single.instance import generate_instance, get_vehicles, map_col
from .coalitions.utilities import encode_coalition
from .coalitions.solutions import nucleolus, shapley
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import h5py

def dict_to_numpy(dictionary : dict):
    """Transform dictionary values into a numpy array."""
    return np.array(list(dictionary.values()))

def compute_nucleolus(players : list, char_function : dict):
    """Compute the nucleolus of a given characteristic function and capture
    runtime errors related to scipy.optimize.linprog errors."""
    try:
        _nucleolus = nucleolus(players, char_function)
        _nucleolus = dict_to_numpy(_nucleolus)
    except RuntimeError:
        _nucleolus = np.zeros(len(players))

    return _nucleolus

def compute_shapley(players : list, char_function : dict):
    """Compute the shapley value of a given characteristic function and
    returns a numpy array instead of a dictionary."""
    _shapley = shapley(players, char_function)
    _shapley = dict_to_numpy(_shapley)

    return _shapley

def generate_observation(n_vehicles : int,
                         n_customers : int,
                         radius : list[float] | float = 0.3,
                         show_plots : bool = False):
    """Compute the relevant utilities for a single instance of the
    collaborative vehicle routing problem.

    Args:
        n_vehicles: integer representing the number of depots in the instance.
        n_customers: integer defining the number of customer nodes in the
        instance.
        radius: float representing the degree of overlap between the customer
        nodes of different vehicles.
        show_plots: boolean determining whether to plot the routes for each
        coalition.

    Returns:
        An instance matrix, a (2**n x n) matrix containing the vehicle assignment
        for each possible coalition, a vector of binary encoded coalitions, a vector
        of the characteristic function, the shapley value and the nucleolus.
    """

    radius = float(np.random.choice(radius)) if isinstance(radius, list) else radius
    instance = generate_instance(n_vehicles, n_customers, radius)
    distance = distances(instance)
    gain, routes = gains(instance, distance)
    vehicles = get_vehicles(instance)
    shap =  compute_shapley(vehicles, gain)
    nucl = compute_nucleolus(vehicles, gain)
    gain = {c: value for c, value in gain.items() if len(c) > 1}
    coalitions = np.array([encode_coalition(list(coalition)) for coalition in gain]).astype(np.int32)
    gain = dict_to_numpy(gain)

    # remove the 'nan' value in the instance matrix
    instance[0, map_col('vehicle')] = -1.0

    return instance, routes, coalitions, gain, shap, nucl

def generate_observations(filename : str = "data/instances.h5",
                          obs : int = 10000,
                          vehicles : int = 3,
                          customers : int = 9,
                          radius : list[float] | float = 0.3,
                          seed : int = 0):
    """Create a h5py dataset to collect all the relevant utilities for a given
    VRP instance.

    The method creates a .h5 dataset of n_obs observations containing
    instance and assignment matrices, and coalition, characteristic
    function, shapley, nucleolus vectors.

    Args:
         filename: a string indicating the desired location of the dataset.
         (by default: data/instance.h5).
         obs: an integer indicating the number of instances in the dataset.
         vehicles: an integer indicating the number of depots per instance.
         customers: an integer indicating the number of customers per instance.
         radius: a float indicating the degree of overlap between the customer
        nodes of different vehicles.
        seed: integer defining the numpy seed affecting the generation.

    Returns:
        Nothing.
    """

    np.random.seed(seed)
    nodes = customers + 1
    coalitions = 2 ** vehicles - (vehicles + 1)
    columns = 5

    with h5py.File(filename, "w") as file:
        instances = file.create_dataset(name = "instance",
                                        shape = (obs, nodes, columns),
                                        dtype = np.float32)
        coalition = file.create_dataset(name = "coalitions",
                                         shape = (obs, coalitions, vehicles),
                                         dtype = np.float32)
        gain = file.create_dataset(name = "gain",
                                   shape = (obs, coalitions),
                                   dtype = np.float32)
        shap = file.create_dataset(name = "shapley",
                                   shape = (obs, vehicles),
                                   dtype = np.float32)
        nucl = file.create_dataset(name = "nucleolus",
                                   shape = (obs, vehicles),
                                   dtype = np.float32)

        for i in tqdm(range(obs)):
            (instances[i],
             _,
             coalition[i],
             gain[i],
             shap[i],
             nucl[i]) = generate_observation(vehicles, customers, radius)