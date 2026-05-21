import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.cluster import KMeans

def distances(Dm : np.ndarray):
    size = len(Dm)
    dist = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            dist[i, j] = np.linalg.norm(Dm[i, 1:3] - Dm[j, 1:3])

    return dist

def clarkeWright(Dm : np.ndarray, 
                 dist : np.ndarray, 
                 size : tuple[int, int],
                 depots : list,
                 colab=None):

    if colab is None:
        colab = []
    N, D = size
    customersID = list(map(int, Dm[D:D+N, 0]))
    routes = {d: None for d in range(D)}

    _Dm = assign(Dm, depots, colab)
    # TODO: one depot, maybe

    for d in depots:

        savings = []
        
        customersID = _Dm[(np.isin(_Dm[:, -1], d) * np.isin(_Dm[:, -2], 0)), 0].astype(int)

        route = {k: [v] for k, v in zip(range(N), customersID)}

        for i, j in itertools.permutations(customersID, 2):
            if i != j:
                saving = dist[d, i] + dist[d, j] - dist[i, j]
                savings.append((i, j, saving))


        savings.sort(key = lambda x: x[-1], reverse = True)

        for i, j, s in savings:
            x = y = None
            for k, v in route.items():
                if v[-1] == i:
                    x = k
                
                if v[0] == j:
                    y = k

            if x is not None and y is not None and x != y:
                new = route[x] + route[y]
                route[x] = new
                del route[y]

        routes[d] = np.array([d] + list(*route.values()) + [d])

    return routes

def plot(Dm, routes, colab):
    node_colors = ['grey', 'red']
    for d in range(2):
        plt.scatter(*Dm[np.isin(Dm[:, -2], d), 1:3].T, c = node_colors[d])
    plt.title(label = f"Instance with colab = {colab}")

    d = 1
    D = len(Dm[np.isin(Dm[:, -2], d)])
    cmap = plt.get_cmap('tab20')
    route_colors = [cmap(i) for i in np.linspace(0, 1, D)]

    for (d, route), color, in zip(routes.items(), route_colors):

        indexes = route.astype(int)
        plt.plot(*Dm[indexes, 1:3].T, label = f'depot {d}', c = color)

    plt.legend()
    plt.show()
    return 0

def assign(Dm, depots : list, colab : list | None = None):

    _Dm = Dm.copy()
    if colab:
        mask = (np.isin(Dm[:, 4], colab) * np.isin(Dm[:, 3], 1))
        kmeans = KMeans(n_clusters = len(colab), init = Dm[mask, 1:3], n_init = 1).fit(Dm[:, 1:3])
        assignment = kmeans.labels_.astype(int)

        for i, d in enumerate(assignment):
            if _Dm[i, -1] in colab and d in colab and _Dm[i, 3] != 1:
                _Dm[i, -1] = d
    return _Dm

def cost(route, dist):
    costs = 0
    for i in range(1, len(route)):
        costs += dist[route[i - 1], route[i]]

    return costs

def gain(routes, pre_routes, dist, coalition):
    _gain = 0
    for depot in coalition:
        _gain += cost(pre_routes[depot], dist) - cost(routes[depot], dist)

    return max(round(_gain, 4), 0)