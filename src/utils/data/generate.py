import numpy as np
import itertools

from utils import clarkeWright, instance, assign, distances, cost, gain
from utils.solutions import shapley, nucleolus


def encode_coalition(combo, D : int):
    encoding = np.zeros(D, dtype = np.int32)
    encoding[list(combo)] = 1
    return encoding

def generate(N, D):

    """
    Based on a slight modification of earlier instance generation code. Despite the inefficiency of this setup, it is simply more convenient.
    """
    _, depots, _, nodes = instance((N, D))
    players = list(range(D))

    # id : id of nodes.
    # pos : x, y position of nodes.
    # depot : binary indicator of depot.
    # vehicle : assignment of nodes to depot/vehicle.
    id = np.arange(N + D).reshape(N + D, 1) 
    pos = np.array(depots + nodes)
    depot = np.concatenate([np.ones((D, 1)), 
                        np.zeros((N, 1))]) 
    vehicle = np.concatenate([np.arange(D).reshape(D, 1),
                    np.random.choice(range(D), size = (N, 1))])
    
    # Create distance matrix of Mak et al. (2023)
    Dm = np.concatenate([id, pos, depot, vehicle], 
                        axis = 1)
    dist = distances(Dm)
    
    # Create variable for characteristic function
    v = {}

    # Compute pre-collaboration routes
    preRoutes = clarkeWright(Dm, dist, (N, D), [0, 1, 2])

    # Store assignment of vehicle to nodes
    A = np.zeros((N + D, 2 ** D), dtype = np.int32)
    coalitions = []
    index = 0

    for size in range(D + 1):
        for combo in itertools.combinations(range(D), size):

            coalitions.append(encode_coalition(combo, D))
            # Store assignment of vehicle to nodes
            _Dm = assign(Dm, [0, 1, 2], list(combo))
            A[:, index] = _Dm[:, -1].copy()
            index += 1

            # Normalize singleton coalitions
            if combo in [(), (0, ), (1, ), (2, )]:
                v[frozenset(combo)] = 0
                continue

            routes = clarkeWright(Dm, 
                                  dist, 
                                  (N, size), 
                                  [0, 1, 2], 
                                  list(combo))
            

            pre = 0
            for depot in combo:
                pre += cost(preRoutes[depot], dist)
            v[frozenset(combo)] = gain(routes, combo, pre, dist)


    try:
        n = nucleolus(players, v).values()
    except:
        n = np.zeros(D)

    try:
        sh = shapley(players, v).values()
    except:
        sh = np.zeros(D)

    coalitions = np.array(coalitions)
    sh = np.array(list(sh))
    n = np.array(list(n))
    v = np.array(list(v.values()))
            
    return Dm, A, coalitions, v, sh, n




    



