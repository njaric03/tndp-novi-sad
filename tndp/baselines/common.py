"""Zajednicke funkcije za baseline algoritme."""

import numpy as np

from tndp.core.assignment import assign, combined_cost
from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork


def network_objective(
    city: CityGraph, network: TransitNetwork, alpha: float = 0.5
) -> tuple[float, float]:
    """Leksikografski cilj: prvo minimalan nepokriven demand, pa cost.

    Izbegava magicne penal konstante: mreza koja pokriva vise demanda je
    uvek bolja, a medju jednako pokrivenim odlucuje kombinovani cost.
    """
    res = assign(city, network, compute_transfers=False)
    return (res.d["d_un"], combined_cost(res, alpha))


def random_route(
    city: CityGraph,
    rng: np.random.Generator,
    min_len: int,
    max_len: int,
    max_tries: int = 50,
) -> list[int] | None:
    """Slucajna prosta putanja u ulicnom grafu duzine u [min_len, max_len].

    Slucajan start pa nasumicno prosirivanje sa oba kraja do ciljne duzine;
    None ako ni posle max_tries pokusaja ne dostigne min_len.
    """
    n = city.n
    finite = np.isfinite(city.street_time) & ~np.eye(n, dtype=bool)
    neighbors = [np.flatnonzero(finite[i]) for i in range(n)]

    for _ in range(max_tries):
        target = int(rng.integers(min_len, max_len + 1))
        route = [int(rng.integers(n))]
        while len(route) < target:
            ends = [route[0], route[-1]]
            options = []
            for side, end in enumerate(ends):
                for c in neighbors[end]:
                    if c not in route:
                        options.append((side, int(c)))
            if not options:
                break
            side, chosen = options[int(rng.integers(len(options)))]
            if side == 0:
                route.insert(0, chosen)
            else:
                route.append(chosen)
        if len(route) >= min_len:
            return route
    return None
