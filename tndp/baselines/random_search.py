"""Random search baseline: najbolja od K nasumicnih validnih mreza."""

import numpy as np

from tndp.baselines.common import network_objective, random_route
from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork


def random_network(
    city: CityGraph,
    num_routes: int,
    min_len: int,
    max_len: int,
    rng: np.random.Generator,
) -> TransitNetwork:
    """Jedna nasumicna mreza od num_routes slucajnih prostih putanja."""
    routes = []
    while len(routes) < num_routes:
        route = random_route(city, rng, min_len, max_len)
        if route is not None:
            routes.append(route)
    return TransitNetwork(routes=routes)


def random_search(
    city: CityGraph,
    num_routes: int,
    min_len: int,
    max_len: int,
    num_samples: int = 1000,
    alpha: float = 0.5,
    seed: int = 0,
) -> tuple[TransitNetwork, tuple[float, float]]:
    """Najbolja od num_samples nasumicnih mreza po (d_un, cost) cilju."""
    rng = np.random.default_rng(seed)
    best_net, best_obj = None, None
    for _ in range(num_samples):
        net = random_network(city, num_routes, min_len, max_len, rng)
        obj = network_objective(city, net, alpha)
        if best_obj is None or obj < best_obj:
            best_net, best_obj = net, obj
    return best_net, best_obj
