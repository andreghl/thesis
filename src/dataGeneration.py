"""
V: number of nodes
N: number of customers
D: number of depots/vehicles
K: number of coalitions (= 2^D)

{
VRP: matrix (Vx5),
A : matrix (V)
coalitions: vector (2^N),
characteristic function: vector (2^N),
nucleolus: matrix (KxN),
shapley: matrix: (KxN)
}
"""

import h5py
import numpy as np

from tqdm import tqdm
from utils.data import generate

runs = 10000
n_customers = 9
n_depots = 3
n_coalitions = 2 ** n_depots
n_nodes = n_depots + n_customers

with h5py.File("data/instances.h5", "w") as f:
        
    Dm = f.create_dataset("Dm", shape = (runs, n_nodes, 5), dtype = np.float32)
    assign = f.create_dataset("assign", shape = (runs, n_nodes, n_coalitions), dtype = np.int32)
    coalitions = f.create_dataset("coalitions", shape = (runs, n_coalitions, n_depots), dtype = np.float32)
    v = f.create_dataset("v", shape = (runs, n_coalitions), dtype = np.float32)
    Sh = f.create_dataset("shapley", (runs, n_depots), dtype = np.float32)
    n = f.create_dataset("nucleolus", (runs, n_depots), dtype = np.float32)

    for i in tqdm(range(runs)):
        Dm[i], assign[i], coalitions[i], v[i], Sh[i], n[i] = generate(n_depots, n_customers)