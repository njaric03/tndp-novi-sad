"""CityGraph: graf grada nad kojim se resava TNDP.

Cvorovi su kandidat lokacije stanica/zona, ulicne ivice nose vreme voznje
u minutima, demand matrica broj putovanja izmedju parova cvorova.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components


@dataclass
class CityGraph:
    """Graf grada. Interni indeksi cvorova su 0-bazirani."""

    coords: np.ndarray       # (n, 2) koordinate cvorova
    street_time: np.ndarray  # (n, n) vreme voznje ulicom u minutima, np.inf gde nema ivice
    demand: np.ndarray       # (n, n) broj putovanja, dijagonala 0
    name: str = ""
    terminal: np.ndarray = field(default=None)  # (n,) bool, sme li linija tu da pocne/zavrsi

    def __post_init__(self):
        if self.terminal is None:
            self.terminal = np.ones(self.n, dtype=bool)

    @property
    def n(self) -> int:
        return self.demand.shape[0]

    @property
    def street_edges(self) -> np.ndarray:
        """Neusmerene ulicne ivice kao (m, 2) niz parova i < j."""
        i, j = np.nonzero(np.isfinite(self.street_time) & ~np.eye(self.n, dtype=bool))
        mask = i < j
        return np.stack([i[mask], j[mask]], axis=1)

    def validate(self) -> list[str]:
        """Vraca listu prekrsaja; prazna lista znaci validan graf."""
        problems = []
        n = self.n
        if self.coords.shape != (n, 2):
            problems.append(f"coords oblik {self.coords.shape}, ocekivano ({n}, 2)")
        if self.street_time.shape != (n, n):
            problems.append(f"street_time oblik {self.street_time.shape}, ocekivano ({n}, {n})")
        if not np.allclose(np.diag(self.demand), 0):
            problems.append("demand dijagonala nije 0")
        if np.any(self.demand < 0):
            problems.append("negativan demand")
        finite = np.isfinite(self.street_time)
        if not np.array_equal(finite, finite.T):
            problems.append("ulicne ivice nisu simetricne")
        elif not np.allclose(
            self.street_time[finite], self.street_time.T[finite]
        ):
            problems.append("vremena voznje nisu simetricna")
        with np.errstate(invalid="ignore"):
            if np.any(self.street_time[finite] < 0):
                problems.append("negativno vreme voznje")
        adj = finite & ~np.eye(n, dtype=bool)
        n_comp, _ = connected_components(adj, directed=False)
        if n_comp != 1:
            problems.append(f"ulicni graf nije povezan ({n_comp} komponenti)")
        return problems

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            coords=self.coords,
            street_time=self.street_time,
            demand=self.demand,
            terminal=self.terminal,
            name=np.array(self.name),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CityGraph":
        data = np.load(path, allow_pickle=False)
        return cls(
            coords=data["coords"],
            street_time=data["street_time"],
            demand=data["demand"],
            terminal=data["terminal"].astype(bool),
            name=str(data["name"]),
        )
