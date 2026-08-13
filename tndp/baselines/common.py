import numpy as np

from tndp.core.assignment import assign, objective
from tndp.core.network import is_duplicate  # noqa: F401  (re-export za baselines)


# tacno ista skalarna funkcija cilja koju RL trenira i po kojoj se izvestava
def network_objective(city, network, scales, alpha=0.5):
    res = assign(city, network, compute_transfers=False)
    return objective(res, scales, alpha)


# cvorovi na kojima linija sme da pocne ili se zavrsi (Mandl instance nose tu masku; kod sintetike su svi cvorovi
def terminal_nodes(city):
    return np.flatnonzero(city.terminal)


# oba kraja linije moraju biti dozvoljeni terminali
def trim_to_terminals(route, city, min_len):
    while len(route) > min_len and not city.terminal[route[0]]:
        route = route[1:]
    while len(route) > min_len and not city.terminal[route[-1]]:
        route = route[:-1]
    if len(route) < min_len or not (city.terminal[route[0]] and city.terminal[route[-1]]):
        return None
    return route


# slucajna prosta putanja u ulicnom grafu: slucajan dozvoljen start pa nasumicno prosirivanje sa oba kraja do ciljne
def random_route(city, rng, min_len, max_len, max_tries=50):
    neighbors = city.neighbors
    starts = terminal_nodes(city)

    for _ in range(max_tries):
        target = int(rng.integers(min_len, max_len + 1))
        route = [int(starts[rng.integers(len(starts))])]
        while len(route) < target:
            options = [(side, int(c)) for side, end in enumerate([route[0], route[-1]])
                       for c in neighbors[end] if c not in route]
            if not options:
                break
            side, chosen = options[int(rng.integers(len(options)))]
            route.insert(0, chosen) if side == 0 else route.append(chosen)
        route = trim_to_terminals(route, city, min_len)
        if route is not None:
            return route
    return None
