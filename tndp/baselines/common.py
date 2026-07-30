import numpy as np

from tndp.core.assignment import assign, normalized_cost


# leksikografski cilj: prvo minimalan nepokriven demand, pa normalizovani
# cost (isti koji RL trenira, pa je poređenje na istom skalaru). scales iz
# cost_scales; leksikografski deo izbegava magične penal konstante
def network_objective(city, network, scales, alpha=0.5):
    res = assign(city, network, compute_transfers=False)
    return (res.d["d_un"], normalized_cost(res, scales, alpha))


# slučajna prosta putanja u uličnom grafu: slučajan start pa nasumično
# proširivanje sa oba kraja do ciljne dužine
def random_route(city, rng, min_len, max_len, max_tries=50):
    n = city.n
    finite = np.isfinite(city.street_time) & ~np.eye(n, dtype=bool)
    neighbors = [np.flatnonzero(finite[i]) for i in range(n)]

    for _ in range(max_tries):
        target = int(rng.integers(min_len, max_len + 1))
        route = [int(rng.integers(n))]
        while len(route) < target:
            options = [(side, int(c)) for side, end in enumerate([route[0], route[-1]])
                       for c in neighbors[end] if c not in route]
            if not options:
                break
            side, chosen = options[int(rng.integers(len(options)))]
            route.insert(0, chosen) if side == 0 else route.append(chosen)
        if len(route) >= min_len:
            return route
    return None
