"""Greedy baseline: iterativno dodavanje linije koja najvise popravlja cilj.

Kandidat linije su najkraci putevi izmedju svih parova cvorova (skraceni
na max_len ako treba), pa se u svakoj od num_routes iteracija bira kandidat
koji leksikografski najvise popravlja (d_un, cost).
"""

import numpy as np
from scipy.sparse.csgraph import dijkstra

from tndp.baselines.common import network_objective
from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork


def shortest_path_candidates(
    city: CityGraph, min_len: int, max_len: int
) -> list[list[int]]:
    """Najkraci ulicni putevi za sve parove, filtrirani na [min_len, max_len]."""
    n = city.n
    street = np.where(np.isfinite(city.street_time), city.street_time, 0.0)
    _, pred = dijkstra(street, directed=False, return_predecessors=True)

    candidates = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            if pred[i, j] < 0 and i != j:
                continue
            path = [j]
            while path[-1] != i:
                path.append(int(pred[i, path[-1]]))
            path.reverse()
            path = path[:max_len]
            if len(path) < min_len:
                continue
            key = tuple(path if path[0] < path[-1] else path[::-1])
            if key not in seen:
                seen.add(key)
                candidates.append(path)
    return candidates


def greedy_network(
    city: CityGraph,
    num_routes: int,
    min_len: int,
    max_len: int,
    alpha: float = 0.5,
) -> tuple[TransitNetwork, tuple[float, float]]:
    """Gradi mrezu dodajuci po jednu najbolju kandidat liniju."""
    candidates = shortest_path_candidates(city, min_len, max_len)
    routes: list[list[int]] = []
    best_obj = None
    for _ in range(num_routes):
        best_cand, step_best = None, None
        for cand in candidates:
            if cand in routes:
                continue
            obj = network_objective(
                city, TransitNetwork(routes=routes + [cand]), alpha
            )
            if step_best is None or obj < step_best:
                best_cand, step_best = cand, obj
        routes.append(best_cand)
        best_obj = step_best
    return TransitNetwork(routes=routes), best_obj
