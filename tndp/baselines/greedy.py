import numpy as np
from scipy.sparse.csgraph import dijkstra

from tndp.baselines.common import is_duplicate, network_objective
from tndp.core.assignment import cost_scales
from tndp.core.network import TransitNetwork


# kandidati: najkraci ulicni putevi svih parova, skraceni na max_len, filtrirani na min_len i na dozvoljene terminale
def shortest_path_candidates(city, min_len, max_len):
    n = city.n
    street = np.where(np.isfinite(city.street_time), city.street_time, 0.0)
    _, pred = dijkstra(street, directed=False, return_predecessors=True)

    candidates, seen = [], set()
    for i in range(n):
        for j in range(i + 1, n):
            if pred[i, j] < 0:
                continue
            path = [j]
            while path[-1] != i:
                path.append(int(pred[i, path[-1]]))
            path.reverse()
            # odsecanje sa obe strane, ne samo sa jedne: put duzi od max_len dao bi uvek isti pocetni komad i time sistematski
            for path in ({tuple(path[:max_len]), tuple(path[-max_len:])}
                         if len(path) > max_len else {tuple(path)}):
                path = list(path)
                if len(path) < min_len:
                    continue
                if not (city.terminal[path[0]] and city.terminal[path[-1]]):
                    continue
                key = tuple(path if path[0] < path[-1] else path[::-1])
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)
    return candidates


# u svakoj iteraciji dodaj kandidata koji najvise popravlja skalarni cilj
def greedy_network(city, num_routes, min_len, max_len, alpha=0.5, evals=None):
    scales = cost_scales(city)
    candidates = shortest_path_candidates(city, min_len, max_len)
    if len(candidates) < num_routes:
        raise ValueError(f"samo {len(candidates)} kandidata za {num_routes} linija")
    routes, best_obj, count = [], None, 0
    for _ in range(num_routes):
        best_cand, step_best = None, None
        for cand in candidates:
            if is_duplicate(cand, routes):
                continue
            obj = network_objective(city, TransitNetwork(routes=routes + [cand]), scales, alpha)
            count += 1
            if step_best is None or obj < step_best:
                best_cand, step_best = cand, obj
        routes.append(best_cand)
        best_obj = step_best
    if evals is not None:
        evals.append(count)
    return TransitNetwork(routes=routes), best_obj
