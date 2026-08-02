import time

import numpy as np

from tndp.baselines.common import (is_duplicate, network_objective,
                                   random_route, terminal_nodes)
from tndp.baselines.greedy import greedy_network
from tndp.core.assignment import cost_scales
from tndp.core.network import TransitNetwork

# Lokalna pretraga nad kompletnim mrežama: ono što u literaturi radi
# metaheuristika (Mumford 2013 SA, Nikolić 2013 BCO, Holliday EA/BCO oko
# naučene politike). Bez ovoga jedini protivnici RL-u su random i
# konstruktivni greedy, koji ne pretražuju uopšte.
#
# Potezi su standardni "make small change" nad jednom linijom:
#   - produži liniju za jednog suseda na nasumičnom kraju,
#   - skrati liniju za jedan čvor sa nasumičnog kraja,
#   - zameni celu liniju novom nasumičnom.
# Prihvata se samo poboljšanje (strogo hill climbing), uz nasumične restarte
# kad pretraga stane.


def _extend(route, city, rng, max_len):
    if len(route) >= max_len:
        return None
    side = int(rng.integers(2))
    end = route[0] if side == 0 else route[-1]
    # dodati čvor postaje novi kraj linije, pa mora biti dozvoljen terminal.
    # to sužava potez (ne može se produžiti "kroz" ne-terminal u jednom
    # koraku), ali potez ostaje ispravan; na sintetici su svi čvorovi
    # terminali pa se ništa ne menja
    options = [int(c) for c in city.neighbors[end]
               if c not in route and city.terminal[c]]
    if not options:
        return None
    node = options[int(rng.integers(len(options)))]
    return [node] + route if side == 0 else route + [node]


def _shorten(route, city, rng, min_len):
    if len(route) <= min_len:
        return None
    cut = route[1:] if int(rng.integers(2)) == 0 else route[:-1]
    # kraj linije mora ostati dozvoljen terminal
    if not (city.terminal[cut[0]] and city.terminal[cut[-1]]):
        return None
    return cut


def _mutate(routes, city, rng, min_len, max_len):
    ri = int(rng.integers(len(routes)))
    move = int(rng.integers(3))
    if move == 0:
        new = _extend(routes[ri], city, rng, max_len)
    elif move == 1:
        new = _shorten(routes[ri], city, rng, min_len)
    else:
        new = random_route(city, rng, min_len, max_len)
    if new is None or is_duplicate(new, routes[:ri] + routes[ri + 1:]):
        return None
    out = [r[:] for r in routes]
    out[ri] = new
    return out


# hill climbing sa restartima. budžet se zadaje brojem evaluacija cilja
# (max_evals) i/ili zidnim vremenom (max_seconds) — oba su potrebna da bi
# poređenje metoda pod istim budžetom uopšte bilo moguće.
# trace: ako se prosledi lista, u nju idu (evaluacija, sekunde, najbolji cilj)
# posle svakog poboljšanja, za anytime krive.
def hill_climb(city, num_routes, min_len, max_len, alpha=0.5, seed=0,
               max_evals=3000, max_seconds=None, init="greedy",
               patience=300, trace=None):
    rng = np.random.default_rng(seed)
    scales = cost_scales(city)
    t0 = time.perf_counter()
    evals = 0

    def score(routes):
        nonlocal evals
        evals += 1
        return network_objective(city, TransitNetwork(routes), scales, alpha)

    def fresh():
        routes = []
        for _ in range(200 * num_routes):  # bez granice bi grad sa malo
            if len(routes) == num_routes:  # terminala mogao da vrti u prazno
                return routes
            r = random_route(city, rng, min_len, max_len)
            if r is not None and not is_duplicate(r, routes):
                routes.append(r)
        raise RuntimeError(f"{city.name}: ne mogu da sastavim {num_routes} "
                           f"različitih linija (dobio {len(routes)})")

    if init == "greedy":
        cur = [r[:] for r in greedy_network(city, num_routes, min_len, max_len,
                                            alpha=alpha)[0].routes]
    else:
        cur = fresh()
    cur_obj = score(cur)
    best, best_obj = [r[:] for r in cur], cur_obj
    if trace is not None:
        trace.append((evals, time.perf_counter() - t0, best_obj))

    stale = 0
    while evals < max_evals:
        if max_seconds is not None and time.perf_counter() - t0 >= max_seconds:
            break
        cand = _mutate(cur, city, rng, min_len, max_len)
        if cand is None:
            continue
        obj = score(cand)
        if obj < cur_obj:
            cur, cur_obj, stale = cand, obj, 0
            if obj < best_obj:
                best, best_obj = [r[:] for r in cand], obj
                if trace is not None:
                    trace.append((evals, time.perf_counter() - t0, best_obj))
        else:
            stale += 1
            if stale >= patience:  # zaglavljeno u lokalnom minimumu, restart
                cur = fresh()
                cur_obj, stale = score(cur), 0

    return TransitNetwork(routes=best), best_obj
