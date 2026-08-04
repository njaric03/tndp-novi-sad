from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import (connected_components, dijkstra,
                                  minimum_spanning_tree)


# graf grada: čvorovi su kandidat stanice/zone, street_time vreme vožnje
# ulicom u minutima (inf gde nema ivice), demand broj putovanja po paru.
# indeksi čvorova su svuda 0-bazirani.
@dataclass
class CityGraph:
    coords: np.ndarray       # (n, 2)
    street_time: np.ndarray  # (n, n)
    demand: np.ndarray       # (n, n), dijagonala 0
    name: str = ""
    terminal: np.ndarray = field(default=None)  # sme li linija tu da počne/završi
    _sp: np.ndarray = field(default=None, repr=False, compare=False)
    _mst: float = field(default=None, repr=False, compare=False)
    _nb: list = field(default=None, repr=False, compare=False)
    _feat: dict = field(default=None, repr=False, compare=False)
    _netfeat: np.ndarray = field(default=None, repr=False, compare=False)
    _edge: tuple = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.terminal is None:
            self.terminal = np.ones(self.n, dtype=bool)

    @property
    def n(self):
        return self.demand.shape[0]

    # scipy nad gustom matricom tretira 0 kao "nema ivice", pa se inf mora
    # prevesti u 0 pre poziva. dijagonala je ionako 0.
    def _street_dense(self):
        return np.where(np.isfinite(self.street_time), self.street_time, 0.0)

    # najkraća vremena ulicom za sve parove; koristi se kao donja granica
    # putničkog troška i za naplatu nepokrivenih parova. keširano jer se
    # zove na svakoj proceni mreže.
    @property
    def street_shortest(self):
        if self._sp is None:
            self._sp = dijkstra(self._street_dense(), directed=False)
        return self._sp

    # susedi po čvoru u uličnom grafu; keširano jer ih env i sve baselines
    # traže na svakom potezu
    @property
    def neighbors(self):
        if self._nb is None:
            finite = np.isfinite(self.street_time) & ~np.eye(self.n, dtype=bool)
            self._nb = [np.flatnonzero(finite[i]) for i in range(self.n)]
        return self._nb

    # donja granica putničkog troška: prosečno ulično vreme po putniku kad
    # bi mreža išla svuda. skala za putnički član funkcije cilja.
    @property
    def street_shortest_mean_demand(self):
        return float((self.demand * self.street_shortest).sum() / self.demand.sum())

    # ukupno vreme minimalnog razapinjućeg stabla: donja granica koliko
    # mreže uopšte treba da bi svaki čvor bio dostupan. služi kao skala
    # operaterskog troška.
    @property
    def mst_time(self):
        if self._mst is None:
            self._mst = float(minimum_spanning_tree(self._street_dense()).sum())
        return self._mst

    @property
    def street_edges(self):
        # neusmerene ivice kao parovi i < j
        i, j = np.nonzero(np.isfinite(self.street_time) & ~np.eye(self.n, dtype=bool))
        mask = i < j
        return np.stack([i[mask], j[mask]], axis=1)

    # lista prekršaja, prazna znači validan graf
    def validate(self):
        problems = []
        n = self.n
        if not np.allclose(np.diag(self.demand), 0) or np.any(self.demand < 0):
            problems.append("demand: dijagonala mora biti 0, vrednosti nenegativne")
        finite = np.isfinite(self.street_time)
        if not np.array_equal(finite, finite.T) or not np.allclose(
            self.street_time[finite], self.street_time.T[finite]
        ):
            problems.append("ulične ivice moraju biti simetrične")
        adj = finite & ~np.eye(n, dtype=bool)
        n_comp, _ = connected_components(adj, directed=False)
        if n_comp != 1:
            problems.append(f"ulični graf nije povezan ({n_comp} komponenti)")
        return problems

    def save(self, path):
        np.savez_compressed(path, coords=self.coords, street_time=self.street_time,
                            demand=self.demand, terminal=self.terminal,
                            name=np.array(self.name))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(coords=d["coords"], street_time=d["street_time"], demand=d["demand"],
                   terminal=d["terminal"].astype(bool), name=str(d["name"]))
