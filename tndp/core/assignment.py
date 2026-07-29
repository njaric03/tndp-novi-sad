"""Passenger assignment: cena mreze iz perspektive putnika i operatera.

Gradi se "route graf" nad platformama (linija, pozicija) plus po jedan
zemaljski cvor po cvoru grada. Ukrcavanje kosta transfer_penalty minuta,
silazak 0, pa put sa k presedanja akumulira (k+1) * penal; prvi penal se
oduzme na kraju. Najkraci putevi se racunaju scipy Dijkstrom nad sparse
matricom, jer se ova funkcija poziva u petlji RL treninga.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork

TRANSFER_PENALTY_MIN = 5.0


@dataclass
class AssignmentResult:
    travel_time: np.ndarray      # (n, n) vreme putovanja sa penalima, np.inf za nepokrivene
    transfers: np.ndarray | None # (n, n) broj presedanja, -1 za nepokrivene; None ako se ne racuna
    C_p: float                   # prosecno vreme putovanja po putniku (pokriveni demand)
    C_o: float                   # ukupno vreme voznje svih linija u jednom smeru
    d: dict                      # d_0, d_1, d_2, d_3p, d_un kao udeli ukupnog demanda

    @property
    def is_connected(self) -> bool:
        return self.d["d_un"] == 0.0


def combined_cost(result: AssignmentResult, alpha: float = 0.5) -> float:
    """Jedina cost funkcija u projektu: C = alpha * C_p + (1 - alpha) * C_o."""
    return alpha * result.C_p + (1.0 - alpha) * result.C_o


def assign(
    city: CityGraph,
    network: TransitNetwork,
    transfer_penalty: float = TRANSFER_PENALTY_MIN,
    compute_transfers: bool = True,
) -> AssignmentResult:
    """Izracuna vremena putovanja, C_p, C_o i d_k statistike za mrezu.

    compute_transfers=False preskace rekonstrukciju puteva (d_k statistike),
    sto je brze i dovoljno za reward u RL treningu.
    """
    n = city.n
    routes = network.routes

    # indeksi platformi: platforma je (linija, pozicija na liniji)
    route_offsets = np.cumsum([0] + [len(r) for r in routes])
    num_platforms = int(route_offsets[-1])
    ground = num_platforms  # zemaljski cvor grada i ima indeks ground + i

    rows, cols, weights = [], [], []
    for ri, route in enumerate(routes):
        base = route_offsets[ri]
        for p, (a, b) in enumerate(zip(route, route[1:])):
            tau = city.street_time[a, b]
            rows += [base + p, base + p + 1]
            cols += [base + p + 1, base + p]
            weights += [tau, tau]
        for p, node in enumerate(route):
            rows += [ground + node, base + p]
            cols += [base + p, ground + node]
            weights += [transfer_penalty, 0.0]

    size = num_platforms + n
    graph = csr_matrix(
        (np.array(weights), (np.array(rows), np.array(cols))), shape=(size, size)
    )

    ground_ids = np.arange(ground, ground + n)
    if compute_transfers:
        dist, pred = dijkstra(
            graph, directed=True, indices=ground_ids, return_predecessors=True
        )
    else:
        dist = dijkstra(graph, directed=True, indices=ground_ids)
        pred = None

    # vreme putovanja: oduzmi penal prvog ukrcavanja
    travel_time = dist[:, ground_ids] - transfer_penalty
    np.fill_diagonal(travel_time, 0.0)

    demand = city.demand
    total_demand = demand.sum()
    served = np.isfinite(travel_time)
    served_demand = demand[served].sum()

    if served_demand > 0:
        C_p = float((demand[served] * travel_time[served]).sum() / served_demand)
    else:
        C_p = float("inf")
    C_o = float(network.route_times(city).sum())

    transfers = None
    d = {"d_0": 0.0, "d_1": 0.0, "d_2": 0.0, "d_3p": 0.0}
    if compute_transfers:
        transfers = _count_transfers(pred, ground_ids, num_platforms, n)
        transfers[~served] = -1
        np.fill_diagonal(transfers, 0)
        if total_demand > 0:
            offdiag = ~np.eye(n, dtype=bool)
            mask0 = served & (transfers == 0) & offdiag
            d["d_0"] = float(demand[mask0].sum() / total_demand)
            d["d_1"] = float(demand[served & (transfers == 1)].sum() / total_demand)
            d["d_2"] = float(demand[served & (transfers == 2)].sum() / total_demand)
            d["d_3p"] = float(demand[served & (transfers >= 3)].sum() / total_demand)
    d["d_un"] = float(demand[~served].sum() / total_demand) if total_demand > 0 else 0.0

    return AssignmentResult(
        travel_time=travel_time, transfers=transfers, C_p=C_p, C_o=C_o, d=d
    )


def _count_transfers(
    pred: np.ndarray, ground_ids: np.ndarray, num_platforms: int, n: int
) -> np.ndarray:
    """Broj presedanja po paru: broj ukrcavanja duz najkraceg puta minus 1.

    Ukrcavanje je prelaz zemaljski cvor -> platforma. Putevi se rekonstruisu
    hodom po predecessor matrici; grafovi su mali pa je python petlja ok.
    """
    transfers = np.full((n, n), -1, dtype=int)
    for si in range(n):
        p_row = pred[si]
        for tj in range(n):
            if si == tj:
                continue
            cur = ground_ids[tj]
            boardings = 0
            while True:
                prev = p_row[cur]
                if prev < 0:
                    break
                if prev >= num_platforms and cur < num_platforms:
                    boardings += 1
                cur = prev
            if cur == ground_ids[si] and boardings > 0:
                transfers[si, tj] = boardings - 1
    return transfers
