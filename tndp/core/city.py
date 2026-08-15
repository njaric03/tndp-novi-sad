from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.sparse.csgraph import (connected_components, dijkstra,
                                  minimum_spanning_tree)


# graf grada: cvorovi su kandidat stanice/zone, street_time vreme voznje ulicom u minutima (inf gde nema ivice)
@dataclass
class CityGraph:
    coords: np.ndarray       # (n, 2)
    street_time: np.ndarray  # (n, n)
    demand: np.ndarray       # (n, n), dijagonala 0
    name: str = ""
    terminal: np.ndarray = field(default=None)  # sme li linija tu da pocne/zavrsi
    # kesevi koje puni features.py; ostalo kesira cached_property nize
    _feat: dict = field(default=None, repr=False, compare=False)
    _netfeat: dict = field(default=None, repr=False, compare=False)
    _edge: tuple = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.terminal is None:
            self.terminal = np.ones(self.n, dtype=bool)

    @property
    def n(self):
        return self.demand.shape[0]

    # scipy nad gustom matricom tretira 0 kao "nema ivice", pa se inf mora prevesti u 0 pre poziva
    def _street_dense(self):
        return np.where(np.isfinite(self.street_time), self.street_time, 0.0)

    # najkraca vremena ulicom za sve parove; koristi se kao donja granica putnickog troska i za naplatu nepokrivenih parova
    @cached_property
    def street_shortest(self):
        return dijkstra(self._street_dense(), directed=False)

    # susedi po cvoru u ulicnom grafu; kesirano jer ih env i sve baselines traze na svakom potezu
    @cached_property
    def neighbors(self):
        finite = np.isfinite(self.street_time) & ~np.eye(self.n, dtype=bool)
        return [np.flatnonzero(finite[i]) for i in range(self.n)]

    @property
    def street_shortest_mean_demand(self):
        return float((self.demand * self.street_shortest).sum() / self.demand.sum())

    # donja granica koliko mreze uopste treba da bi svaki cvor bio dostupan
    @cached_property
    def mst_time(self):
        return float(minimum_spanning_tree(self._street_dense()).sum())

    @cached_property
    def street_edges(self):
        # neusmerene ivice kao parovi i < j
        i, j = np.nonzero(np.isfinite(self.street_time) & ~np.eye(self.n, dtype=bool))
        mask = i < j
        return np.stack([i[mask], j[mask]], axis=1)

    # lista prekrsaja, prazna znaci validan graf
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

    # isto sto i validate(), ali puca; skripte hoce da stanu na neispravnom ulazu
    def require_valid(self):
        problems = self.validate()
        if problems:
            raise ValueError(f"nevalidan grad {self.name}: {problems}")

    def save(self, path):
        np.savez_compressed(path, coords=self.coords, street_time=self.street_time,
                            demand=self.demand, terminal=self.terminal,
                            name=np.array(self.name))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(coords=d["coords"], street_time=d["street_time"], demand=d["demand"],
                   terminal=d["terminal"].astype(bool), name=str(d["name"]))
