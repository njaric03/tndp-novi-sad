import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

from tndp.core.assignment import BUS_SPEED_KMH  # deljeno sa cost funkcijom
from tndp.core.city import CityGraph


def _connected(adj):
    return connected_components(adj, directed=False)[0] == 1


# sintetički grad: random tačke, Delaunay ivice, proređivanje do realistične
# gustine ulica. demand_mode "uniform" replicira Holliday U[60, 800] režim,
# "gravity" daje prostornu strukturu preko masa čvorova (naš glavni režim).
def generate_city(n=None, seed=0, demand_mode="gravity", n_range=(20, 60),
                  edge_keep=0.85, beta=2.0):
    rng = np.random.default_rng(seed)
    if n is None:
        n = int(rng.integers(n_range[0], n_range[1] + 1))
    side = 1.2 * np.sqrt(n)  # km; gustina čvorova ne zavisi od n
    coords = rng.uniform(0, side, (n, 2))

    # delaunay pa izbaci predugačke ivice (artefakti konveksnog omotača)
    edges = set()
    for simplex in Delaunay(coords).simplices:
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            i, j = sorted((simplex[a], simplex[b]))
            edges.add((int(i), int(j)))
    edges = list(edges)
    lengths = {e: float(np.linalg.norm(coords[e[0]] - coords[e[1]])) for e in edges}
    med = np.median(list(lengths.values()))

    adj = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        adj[i, j] = adj[j, i] = True

    # izbacivanje dugih ivica i proređivanje idu kroz istu proveru: ivica se
    # sme skinuti samo ako graf ostane povezan. ranije su duge ivice padale
    # bez provere, pa je ~1% gradova izlazio nepovezan — a nepovezan grad
    # daje beskonačnu donju granicu putničkog troška, što je u staroj
    # funkciji cilja tiho gasilo ceo putnički član (deljenje sa inf).
    def drop_if_safe(i, j):
        adj[i, j] = adj[j, i] = False
        if not _connected(adj):
            adj[i, j] = adj[j, i] = True

    for e in sorted(edges, key=lambda e: -lengths[e]):
        if lengths[e] > 2.5 * med:
            drop_if_safe(*e)
    for e in sorted(edges, key=lambda e: rng.random()):
        if adj[e[0], e[1]] and rng.random() >= edge_keep:
            drop_if_safe(*e)

    dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    street = np.where(adj, dist / BUS_SPEED_KMH * 60, np.inf)  # minuti
    np.fill_diagonal(street, 0.0)

    # demand: ukupan broj putovanja isti u oba režima da budu uporedivi
    total_trips = 430.0 * n * (n - 1)
    if demand_mode == "uniform":
        demand = rng.uniform(60, 800, (n, n))
    else:
        # gravity: mase čvorova (produkcija i atrakcija), opadanje sa daljinom
        prod = rng.lognormal(0, 0.8, n)
        attr = rng.lognormal(0, 0.8, n)
        with np.errstate(divide="ignore"):
            demand = prod[:, None] * attr[None, :] / np.maximum(dist, 0.3) ** beta
        demand *= rng.lognormal(0, 0.3, (n, n))  # šum
    demand = (demand + demand.T) / 2
    np.fill_diagonal(demand, 0.0)
    demand *= total_trips / demand.sum()

    city = CityGraph(coords=coords, street_time=street, demand=demand,
                     name=f"synth-{demand_mode}-{seed}")
    problems = city.validate()
    if problems:  # nikad ne vraćaj grad koji ne prolazi sopstvenu validaciju
        raise RuntimeError(f"generator dao nevalidan grad (seed={seed}): {problems}")
    return city


# brz vizuelni pregled: python -m tndp.synth
if __name__ == "__main__":
    import time
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    t0 = time.perf_counter()
    for k, ax in enumerate(axes.flat):
        city = generate_city(seed=k, demand_mode="gravity" if k < 5 else "uniform")
        for i, j in city.street_edges:
            ax.plot(city.coords[[i, j], 0], city.coords[[i, j], 1],
                    color="gray", lw=0.7, zorder=1)
        w = city.demand.sum(axis=0) + city.demand.sum(axis=1)
        ax.scatter(city.coords[:, 0], city.coords[:, 1],
                   s=8 + 80 * w / w.max(), color="tab:blue", zorder=2)
        ax.set_title(f"{city.name} (n={city.n})", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    dt = time.perf_counter() - t0
    print(f"10 gradova za {dt:.2f} s")

    out = Path(__file__).parent.parent / "results" / "synth-preview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"snimljeno u {out}")
