import numpy as np

def tweak(candidate : np.ndarray):
    operators = []
    # need to add the operators without the brackets if they must take an argument.
    operator =  np.random.choice(operators)
    return operator(candidate)

def cost(candidate, dist : np.ndarray, depots : list, coalition : list):

    for depot in candidate:
        a = 1
    return 0

def assign_customers(dm : np.ndarray, dist : np.ndarray, depots : list, coalition : list | None):
    """
    This method seeks to determine an optimal assigment of customers to vehicles for the Clarke Wright savings
    algorithm using Local Search.
    """

    if coalition is None:
        coalition = []

    _dm = dm.copy()
    vehicle_index = -1
    initial = dm[:, vehicle_index].astype(int)

    weights = dist[:, 0:len(depots)]
    weights_sum = weights.sum(axis = 1)[:, None]
    weights = (weights / weights_sum).round(4)

    assignment = np.argmin(weights, axis = 1)

    n = assignment.shape[0]
    for i in range(n):
        if assignment[i] not in coalition or initial[i] not in coalition:
            assignment[i] = initial[i]

    _dm[:, vehicle_index] = assignment

    return _dm