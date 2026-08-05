from pathlib import Path

import numpy as np

from tndp.core.city import CityGraph


# instance u formatu repoa RenatoArbex/TransitNetworkDesign: tri CSV fajla ({prefix}_nodes/links/demand.txt)
def load_benchmark_city(instance_dir, name=None):
    instance_dir = Path(instance_dir)
    prefix = instance_dir.name.lower()
    name = name or instance_dir.name

    nodes = np.genfromtxt(instance_dir / f"{prefix}_nodes.txt", delimiter=",", skip_header=1)
    links = np.genfromtxt(instance_dir / f"{prefix}_links.txt", delimiter=",", skip_header=1)
    dem = np.genfromtxt(instance_dir / f"{prefix}_demand.txt", delimiter=",", skip_header=1)

    n = nodes.shape[0]
    street = np.full((n, n), np.inf)
    np.fill_diagonal(street, 0.0)
    street[links[:, 0].astype(int) - 1, links[:, 1].astype(int) - 1] = links[:, 2]
    street = np.minimum(street, street.T)  # fajlovi navode oba smera, za svaki slučaj

    demand = np.zeros((n, n))
    demand[dem[:, 0].astype(int) - 1, dem[:, 1].astype(int) - 1] = dem[:, 2]
    demand = np.maximum(demand, demand.T)
    np.fill_diagonal(demand, 0.0)

    terminal = nodes[:, 3].astype(bool) if nodes.shape[1] > 3 else None
    return CityGraph(coords=nodes[:, 1:3], street_time=street, demand=demand,
                     name=name, terminal=terminal)


# fajl sa rešenjima iz literature: ime, broj linija R, pa R redova "1-2-3"
def load_literature_solutions(path):
    solutions = {}
    for block in Path(path).read_text().split("\n\n"):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        num = int(lines[1])
        solutions[lines[0]] = [[int(x) - 1 for x in ln.split("-")]
                               for ln in lines[2:2 + num]]
    return solutions
