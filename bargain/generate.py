from .coalitions.players import get_players
from .coalitions.solutions import nucleolus, shapley
from .coalitions.utilities import get_coalitions_encode, yield_coalitions
from .routing.instance import generate_instance
from .routing.heuristics import clarke_wright, assign_customers
from .routing.utilities import distances, gain, plot
from tqdm import tqdm

import h5py
import numpy as np
import matplotlib.pyplot as plt


def get_assignment(instance : np.ndarray):
    """Return the last column of the instance matrix containing vehicle assignments."""
    vehicle_col = 4
    return instance[:, vehicle_col]

def dict_to_numpy(dictionary : dict):
    """Transform dict values into a numpy array."""
    return np.array(list(dictionary.values()))

def compute_nucleolus(players : list, char_function : dict):
    """Compute the nucleolus of a given characteristic function and capture
    runtime errors related to scipy.optimize.linprog errors."""
    n_players = len(players)
    try:
        _nucleolus = nucleolus(players, char_function)
        _nucleolus = dict_to_numpy(_nucleolus)

    except RuntimeError:
        _nucleolus = np.zeros(n_players)

    return _nucleolus

def compute_shapley(players : list, char_function : dict):
    """Compute the shapley value of a given characteristic function and
    returns a numpy array instead of a dictionary."""
    _shapley = shapley(players, char_function)
    _shapley = dict_to_numpy(_shapley)

    return _shapley

def generate_observation(n_depots : int,
                         n_customers : int,
                         radius : list[float] | float = 0.3,
                         show_plots : bool = False):
    """Compute the relevant utilities for a single instance of the
    collaborative vehicle routing problem.

    Args:
        n_depots: integer representing the number of depots in the instance.
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
    if isinstance(radius, float):
        _radius = radius
    else:
        _radius = float(np.random.choice(radius))

    n_coalitions = 2 ** n_depots
    n_nodes = n_depots + n_customers
    instance = generate_instance(n_depots, n_customers, _radius)
    distance = distances(instance)
    players = get_players(instance)
    char_function = {}
    assignments = np.zeros((n_nodes, n_coalitions), dtype = np.int32)

    # compute the routing distance based on the initial assignment
    initial_routes = clarke_wright(instance, distance)

    for i, coalition in enumerate(yield_coalitions(players)):

        _instance = assign_customers(instance, distance, coalition)
        assignments[:, i] = get_assignment(_instance)

        if len(coalition) < 2:
            char_function[frozenset(coalition)] = 0.0
            continue

        colab_routes = clarke_wright(_instance, distance)
        char_function[frozenset(coalition)] = gain(colab_routes, initial_routes, distance)

        if show_plots:
            fig, (left, right) = plt.subplots(nrows = 1, ncols = 2)
            plot(instance, routes = initial_routes, coalition = [], ax = left)
            plot(_instance, routes = colab_routes, coalition = coalition, ax = right)
            plt.show()

    shap = compute_shapley(players, char_function)
    nucl = compute_nucleolus(players, char_function)
    char = dict_to_numpy(char_function)
    coal = get_coalitions_encode(n_depots)

    return instance, assignments, coal, char, shap, nucl

def generate_observations(filename : str = "data/instances.h5",
                          n_obs : int = 10000,
                          n_depots : int = 3,
                          n_customers : int = 9,
                          radius : list[float] | float = 0.3,
                          seed : int = 0,
                          print_mode : bool = False):
    """Create a h5py dataset to collect all the relevant utilities for a given
    VRP instance.

    The method creates a .h5 dataset of n_obs observations containing
    instance and assignment matrices, and coalition, characteristic
    function, shapley, nucleolus vectors.

    Args:
         filename: a string indicating the desired location of the dataset.
         (by default: data/instance.h5).
         n_obs: an integer indicating the number of instances in the dataset.
         n_depots: an integer indicating the number of depots per instance.
         n_customers: an integer indicating the number of customers per instance.
         radius: a float indicating the degree of overlap between the customer
        nodes of different vehicles.
        seed: integer defining the numpy seed affecting the generation.
        print_mode: boolean determining whether the generated instances are printed
        to the terminal.

    Returns:
        Nothing.
    """
    np.random.seed(seed)
    n_nodes = n_depots + n_customers
    n_coalitions = 2 ** n_depots
    instance_cols = 5

    with h5py.File(filename, "w") as file:
        instances = file.create_dataset(name = "instance",
                                        shape = (n_obs, n_nodes, instance_cols),
                                        dtype = np.float32)
        assignments = file.create_dataset(name = "assignments",
                                          shape = (n_obs, n_nodes, n_coalitions),
                                          dtype = np.int32)
        coalitions = file.create_dataset(name = "coalitions",
                                         shape = (n_obs, n_coalitions, n_depots),
                                         dtype = np.float32)
        char_functions = file.create_dataset(name = "char_function",
                                             shape = (n_obs, n_coalitions),
                                             dtype = np.float32)
        shap = file.create_dataset(name = "shapley",
                                   shape = (n_obs, n_depots),
                                   dtype = np.float32)
        nucl = file.create_dataset(name = "nucleolus",
                                   shape = (n_obs, n_depots),
                                   dtype = np.float32)

        for i in tqdm(range(n_obs)):
            (instances[i],
             assignments[i],
             coalitions[i],
             char_functions[i],
             shap[i],
             nucl[i]) = generate_observation(n_depots, n_customers, radius)

            if print_mode:
                print(instances[i],
                      assignments[i],
                      coalitions[i],
                      char_functions[i],
                      shap[i],
                      nucl[i])