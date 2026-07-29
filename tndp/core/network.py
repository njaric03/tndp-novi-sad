"""TransitNetwork: skup linija nad CityGraph-om i provera ogranicenja."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tndp.core.city import CityGraph


@dataclass
class TransitNetwork:
    """Lista linija; svaka linija je lista 0-baziranih indeksa cvorova."""

    routes: list[list[int]]

    def check(
        self,
        city: CityGraph,
        num_routes: int | None = None,
        min_len: int = 2,
        max_len: int | None = None,
    ) -> list[str]:
        """Vraca listu prekrsaja ogranicenja; prazna lista znaci validna mreza.

        Povezanost mreze se ne proverava ovde nego kroz assignment
        (d_un > 0 znaci nepokrivene parove), da bi cost i validnost
        dolazili iz istog prolaza.
        """
        problems = []
        if num_routes is not None and len(self.routes) != num_routes:
            problems.append(f"broj linija {len(self.routes)}, trazeno {num_routes}")
        for ri, route in enumerate(self.routes):
            if len(route) < min_len:
                problems.append(f"linija {ri}: duzina {len(route)} < {min_len}")
            if max_len is not None and len(route) > max_len:
                problems.append(f"linija {ri}: duzina {len(route)} > {max_len}")
            if len(set(route)) != len(route):
                problems.append(f"linija {ri}: ponovljen cvor")
            if any(not 0 <= v < city.n for v in route):
                problems.append(f"linija {ri}: indeks cvora van opsega")
                continue
            for a, b in zip(route, route[1:]):
                if not np.isfinite(city.street_time[a, b]) or a == b:
                    problems.append(f"linija {ri}: ivica {a}-{b} ne postoji u ulicnom grafu")
        return problems

    def route_times(self, city: CityGraph) -> np.ndarray:
        """Vreme voznje svake linije u jednom smeru, u minutima."""
        return np.array([
            sum(city.street_time[a, b] for a, b in zip(route, route[1:]))
            for route in self.routes
        ])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"routes": self.routes}))

    @classmethod
    def load(cls, path: str | Path) -> "TransitNetwork":
        return cls(routes=json.loads(Path(path).read_text())["routes"])
