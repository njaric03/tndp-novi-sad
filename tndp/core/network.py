import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# kanonski oblik linije: linija i njen obrnuti redosled su ista linija, pa
# se provera na duplikate mora raditi preko ovoga a ne preko `route in routes`
def canon(route):
    return tuple(min(route, route[::-1]))


def is_duplicate(route, routes):
    return canon(route) in {canon(r) for r in routes}


# mreža linija; svaka linija je lista indeksa čvorova
@dataclass
class TransitNetwork:
    routes: list

    # prekršaji ograničenja; povezanost se ne proverava ovde nego kroz
    # assignment (d_un > 0), da validnost i cost dolaze iz istog prolaza
    def check(self, city, num_routes=None, min_len=2, max_len=None):
        problems = []
        if num_routes is not None and len(self.routes) != num_routes:
            problems.append(f"broj linija {len(self.routes)}, traženo {num_routes}")
        # linija i njen obrnuti redosled su ista linija
        keys = [canon(r) for r in self.routes if r]
        if len(set(keys)) != len(keys):
            problems.append("mreža sadrži duplirane linije")
        for ri, route in enumerate(self.routes):
            if not route:
                problems.append(f"linija {ri}: prazna")
                continue
            if len(route) < min_len or (max_len is not None and len(route) > max_len):
                problems.append(f"linija {ri}: dužina {len(route)} van [{min_len}, {max_len}]")
            if len(set(route)) != len(route):
                problems.append(f"linija {ri}: ponovljen čvor")
            if not (city.terminal[route[0]] and city.terminal[route[-1]]):
                problems.append(f"linija {ri}: kraj nije dozvoljen terminal")
            for a, b in zip(route, route[1:]):
                if not np.isfinite(city.street_time[a, b]) or a == b:
                    problems.append(f"linija {ri}: ivica {a}-{b} ne postoji u uličnom grafu")
        return problems

    # vreme vožnje svake linije u jednom smeru
    def route_times(self, city):
        return np.array([sum(city.street_time[a, b] for a, b in zip(r, r[1:]))
                         for r in self.routes])

    def save(self, path):
        Path(path).write_text(json.dumps({"routes": self.routes}))

    @classmethod
    def load(cls, path):
        return cls(routes=json.loads(Path(path).read_text())["routes"])
