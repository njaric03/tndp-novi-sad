import time

import numpy as np

from tndp.baselines.routes import is_duplicate, random_route
from tndp.core.assignment import cost_scales, network_objective
from tndp.core.network import TransitNetwork


def random_network(city, num_routes, min_len, max_len, rng):
    routes = []
    while len(routes) < num_routes:
        route = random_route(city, rng, min_len, max_len)
        if route is not None and not is_duplicate(route, routes):
            routes.append(route)
    return TransitNetwork(routes=routes)


def random_search(city, num_routes, min_len, max_len, num_samples=1000,
                  alpha=0.5, seed=0, trace=None):
    rng = np.random.default_rng(seed)
    scales = cost_scales(city)
    t0 = time.perf_counter()
    best_net, best_obj = None, None
    for k in range(num_samples):
        net = random_network(city, num_routes, min_len, max_len, rng)
        obj = network_objective(city, net, scales, alpha)
        if best_obj is None or obj < best_obj:
            best_net, best_obj = net, obj
            if trace is not None:
                trace.append((k + 1, time.perf_counter() - t0, best_obj))
    return best_net, best_obj
