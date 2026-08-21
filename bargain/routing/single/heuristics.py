from .instance import map_col, get_depot, get_vehicles, generate_instance

import numpy as np
import itertools


def get_demand(route : list):
    """Return the demand on a specific route."""
    return len(route)

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

    n, _ = instance.shape
    distance = np.zeros((n, n), dtype = np.float64)

    for i in range(n):
        for j in range(n):
            distance[i, j] = np.linalg.norm(instance[i, 1:3] - instance[j, 1:3])

    return distance

def get_capacities(instance : np.ndarray):
    """Return the capacity of each vehicle based on the allocation in the initial VRP."""
    vehicles, counts = get_vehicles(instance, return_counts = True)
    return {v: c for v, c in zip(vehicles, counts)}

def get_customers_id(instance : np.ndarray, coalition : list | int):
    """Return the set of customers assigned to the vehicle in the coalition."""
    mask = np.isin(instance[:, map_col('vehicle')], test_elements = coalition)
    return instance[mask, map_col('id')]

def _assign_routes(instance : np.ndarray, routes : dict, coalition : list[int]):
    """"""
    overlap = {v: [] for v in coalition}
    maxim = {v: 0 for v in coalition}
    valid = {v: [] for v in coalition}

    for v in coalition:
        customers = get_customers_id(instance, v)
        overlap[v] = [len(set(customers) & set(routes[k])) for k in routes]
        maxim[v] = max(overlap[v])

    indices = np.argsort(-np.array(list(maxim.values())))

    for i in indices:
        v = list(coalition)[i]
        k = int(np.argmax(overlap[v]))

        valid[list(coalition)[i]] = routes[list(coalition)[k]].copy()

        for j in coalition:
            overlap[j][k] = 0

    return valid

def assign_routes(instance : np.ndarray, routes : dict, coalition : list[int]):
    """"""
    valid = {v: [] for v in coalition}
    cap = get_capacities(instance)
    usage = {v: cap[v] for v in coalition}

    segments = list(routes.values())
    segments.sort(key = lambda s: len(s), reverse = True)

    for segment in segments:
        demand = len(segment)
        assigned = False
        for v in coalition:
            # Check if vehicle can take this segment
            if demand <= usage[v]:
                valid[v] = segment.copy()
                usage[v] -= demand
                assigned = True
                break

        if not assigned:
            print(f'segment {segment} not assigned')
            pass

    return valid

def clarke_wright(instance : np.ndarray,
                  distance : np.ndarray,
                  coalition : list):
    """Return a set of routes for k vehicles starting from one depot on
    instance with n nodes (and n - 1 customers).

    Args:
        instance: a (n x 5) vehicle routing instance matrix.
        distance: a (n x n) distance matrix.
        coalition: a list of vehicle ids in the collaboration scheme.

    Returns:
        a dictionary of list for each vehicle representing
        the route taken by that vehicle.
    """

    depot = get_depot(instance)[0, map_col('id')].astype(int)
    vehicles = get_vehicles(instance)
    cap = get_capacities(instance)

    savings = []

    # identify the customers of the vehicle under consideration.
    customers = get_customers_id(instance, coalition).astype(int)

    routes = {k: [v] for k, v in enumerate(customers)}

    for i, j in itertools.permutations(customers, r = 2):
        if i != j:
            save = distance[depot, i] + distance[depot, j] - distance[i, j]
            savings.append((i, j, save))

    savings.sort(key = lambda x : x[-1], reverse = True)
    valid = {v: [] for v in coalition}

    for i, j, s in savings:
        x = y = None
        for k, v in routes.items():
            if v[-1] == i:
                x = k

            if v[0] == j:
                y = k

        if x is not None and y is not None and x != y:
            new_route = routes[x] + routes[y]

            for vehicle in coalition:
                if (get_demand(new_route) <= cap[vehicle]
                        and set(valid[vehicle]) < set(new_route)):
                    valid[vehicle] = new_route.copy()

                    routes[x] = new_route
                    del routes[y]
                    break


    # {v: route for v, route in zip(coalition, routes.values())}
    routes = assign_routes(instance, valid, coalition)

    return {v: [depot] + route + [depot] for v, route in routes.items()}

def _clarke_wright(instance : np.ndarray,
                  distance : np.ndarray,
                  coalition : list):
    """Return a set of routes for k vehicles starting from one depot on
    instance with n nodes (and n - 1 customers).

    Args:
        instance: a (n x 5) vehicle routing instance matrix.
        distance: a (n x n) distance matrix.
        coalition: a list of vehicle ids in the collaboration scheme.

    Returns:
        a dictionary of list for each vehicle representing
        the route taken by that vehicle.
    """
    depot = get_depot(instance)[0, map_col('id')].astype(int)
    # vehicles = get_vehicles(instance) # Unused
    cap = get_capacities(instance)

    savings = []

    # Identify the customers of the coalition
    customers = get_customers_id(instance, coalition).astype(int)

    # Initialize routes: each customer is its own route
    # Key: route_id (integer index), Value: list of customer IDs
    routes = {k: [v] for k, v in enumerate(customers)}

    # Calculate savings for all pairs
    for i, j in itertools.permutations(customers, r=2):
        if i != j:
            save = distance[depot, i] + distance[depot, j] - distance[i, j]
            savings.append((i, j, save))

    # Sort savings descending
    savings.sort(key=lambda x: x[-1], reverse=True)

    # Merge routes based on savings
    # We do NOT assign to vehicles yet. We just merge the customer lists.
    for i, j, s in savings:
        x = y = None

        # Find which route ends with i and which starts with j
        for k, v in routes.items():
            if not v: continue
            if v[-1] == i:
                x = k
            if v[0] == j:
                y = k

        # Merge if both found and they are different routes
        if x is not None and y is not None and x != y:
            new_route = routes[x] + routes[y]

            # Update the routes dictionary
            routes[x] = new_route
            routes[y] = [] # Mark as deleted/empty

            # Clean up the dictionary to remove empty entries
            # (Optional but keeps the list clean)
            # routes = {k: v for k, v in routes.items() if v}

    # --- PHASE 2: Assign Merged Routes to Vehicles (Bin Packing) ---

    # Get the list of actual merged routes (non-empty)
    merged_segments = [r for r in routes.values() if r]

    # Sort segments by demand (length) descending - "Largest First" heuristic
    merged_segments.sort(key=len, reverse=True)

    # Initialize valid routes for each vehicle in the coalition
    valid = {v: [] for v in coalition}

    # Track remaining capacity for each vehicle
    # We create a local copy to track usage during assignment
    remaining_cap = {v: cap[v] for v in coalition}

    for segment in merged_segments:
        demand = len(segment)
        assigned = False

        # Try to assign this segment to a vehicle with enough capacity
        for v in coalition:
            if remaining_cap[v] >= demand:
                valid[v] = segment
                remaining_cap[v] -= demand
                assigned = True
                break

        if not assigned:
            # If no vehicle can take this segment, the coalition is infeasible.
            # In a strict solver, you might raise an error or split the segment.
            # Here we leave it unassigned, which will result in a high cost
            # or missing customers in the final gain calculation.
            pass

    print([len(route) for route in valid.values()])

    # Add depot to start and end of each assigned route
    return {v: [depot] + route + [depot] for v, route in valid.items()}