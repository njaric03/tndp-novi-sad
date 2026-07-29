"""Ucitavanje benchmark instanci i resenja iz literature.

Format instanci prati repo RenatoArbex/TransitNetworkDesign: po instanci
tri CSV fajla ({prefix}_nodes.txt, {prefix}_links.txt, {prefix}_demand.txt)
sa 1-baziranim id-jevima cvorova. Vremena su u minutima.
"""

from pathlib import Path

import numpy as np

from tndp.core.city import CityGraph


def load_benchmark_city(instance_dir: str | Path, name: str | None = None) -> CityGraph:
    """Ucita instancu iz foldera, npr. data/benchmarks/Mandl/Mandl1."""
    instance_dir = Path(instance_dir)
    prefix = instance_dir.name.lower()
    if name is None:
        name = instance_dir.name

    nodes = np.genfromtxt(
        instance_dir / f"{prefix}_nodes.txt", delimiter=",", skip_header=1
    )
    links = np.genfromtxt(
        instance_dir / f"{prefix}_links.txt", delimiter=",", skip_header=1
    )
    demand_rows = np.genfromtxt(
        instance_dir / f"{prefix}_demand.txt", delimiter=",", skip_header=1
    )

    n = nodes.shape[0]
    ids = nodes[:, 0].astype(int)
    if not np.array_equal(ids, np.arange(1, n + 1)):
        raise ValueError(f"{name}: id-jevi cvorova nisu 1..n")

    coords = nodes[:, 1:3]
    terminal = nodes[:, 3].astype(bool) if nodes.shape[1] > 3 else None

    street_time = np.full((n, n), np.inf)
    np.fill_diagonal(street_time, 0.0)
    src = links[:, 0].astype(int) - 1
    dst = links[:, 1].astype(int) - 1
    street_time[src, dst] = links[:, 2]
    # fajlovi navode ivice u oba smera, ali za svaki slucaj simetrizujemo
    street_time = np.minimum(street_time, street_time.T)

    demand = np.zeros((n, n))
    dsrc = demand_rows[:, 0].astype(int) - 1
    ddst = demand_rows[:, 1].astype(int) - 1
    demand[dsrc, ddst] = demand_rows[:, 2]
    demand = np.maximum(demand, demand.T)
    np.fill_diagonal(demand, 0.0)

    return CityGraph(
        coords=coords, street_time=street_time, demand=demand,
        name=name, terminal=terminal,
    )


def load_literature_solutions(path: str | Path) -> dict[str, list[list[int]]]:
    """Parsira fajl sa resenjima iz literature (literature_solutions_*.txt).

    Format po sekciji: red sa imenom, red sa brojem linija R, pa R redova
    oblika "1-2-3" (1-bazirani id-jevi). Sekcije razdvaja prazan red.
    Vraca mapu ime -> lista linija sa 0-baziranim indeksima.
    """
    text = Path(path).read_text()
    solutions: dict[str, list[list[int]]] = {}
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        name = lines[0]
        num_routes = int(lines[1])
        routes = [
            [int(x) - 1 for x in ln.split("-")] for ln in lines[2 : 2 + num_routes]
        ]
        if len(routes) != num_routes:
            raise ValueError(f"sekcija '{name}': najavljeno {num_routes} linija, nadjeno {len(routes)}")
        solutions[name] = routes
    return solutions
