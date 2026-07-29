import numpy as np

from tndp.baselines.common import network_objective, random_route
from tndp.core.network import TransitNetwork


def random_network(city, num_routes, min_len, max_len, rng):
    routes = []
    while len(routes) < num_routes:
        route = random_route(city, rng, min_len, max_len)
        if route is not None:
            routes.append(route)
    return TransitNetwork(routes=routes)


# najbolja od num_samples nasumičnih mreža
def random_search(city, num_routes, min_len, max_len, num_samples=1000, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    best_net, best_obj = None, None
    for _ in range(num_samples):
        net = random_network(city, num_routes, min_len, max_len, rng)
        obj = network_objective(city, net, alpha)
        if best_obj is None or obj < best_obj:
            best_net, best_obj = net, obj
    return best_net, best_obj
