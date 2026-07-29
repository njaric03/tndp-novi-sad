from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# standardni transfer penal iz literature
TRANSFER_PENALTY_MIN = 5.0


@dataclass
class AssignmentResult:
    travel_time: np.ndarray  # (n, n), inf za nepokrivene parove
    transfers: np.ndarray    # (n, n), -1 za nepokrivene; None ako se ne računa
    C_p: float               # prosečno vreme putovanja po putniku
    C_o: float               # ukupno vreme vožnje linija u jednom smeru
    d: dict                  # d_0, d_1, d_2, d_3p, d_un udeli demanda

    @property
    def is_connected(self):
        return self.d["d_un"] == 0.0


# jedina cost funkcija u projektu
def combined_cost(result, alpha=0.5):
    return alpha * result.C_p + (1.0 - alpha) * result.C_o


# passenger assignment preko "route grafa": platforma = (linija, pozicija),
# plus zemaljski čvor po čvoru grada. ukrcavanje košta transfer_penalty,
# silazak 0, pa put sa k presedanja plati (k+1) penala; prvi se oduzme na
# kraju. compute_transfers=False preskače rekonstrukciju puteva (d_k
# statistike), dovoljno i duplo brže za RL reward.
def assign(city, network, transfer_penalty=TRANSFER_PENALTY_MIN, compute_transfers=True):
    n = city.n
    routes = network.routes
    offsets = np.cumsum([0] + [len(r) for r in routes])
    num_platforms = int(offsets[-1])
    ground = num_platforms  # zemaljski čvor grada i ima indeks ground + i

    rows, cols, weights = [], [], []
    for ri, route in enumerate(routes):
        base = offsets[ri]
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
    graph = csr_matrix((np.array(weights), (np.array(rows), np.array(cols))),
                       shape=(size, size))

    ground_ids = np.arange(ground, ground + n)
    if compute_transfers:
        dist, pred = dijkstra(graph, directed=True, indices=ground_ids,
                              return_predecessors=True)
    else:
        dist = dijkstra(graph, directed=True, indices=ground_ids)

    travel_time = dist[:, ground_ids] - transfer_penalty  # skini prvi penal
    np.fill_diagonal(travel_time, 0.0)

    demand = city.demand
    total = demand.sum()
    served = np.isfinite(travel_time)
    served_demand = demand[served].sum()
    C_p = float((demand[served] * travel_time[served]).sum() / served_demand) \
        if served_demand > 0 else float("inf")
    C_o = float(network.route_times(city).sum())

    transfers = None
    d = {"d_0": 0.0, "d_1": 0.0, "d_2": 0.0, "d_3p": 0.0}
    if compute_transfers:
        transfers = _count_transfers(pred, ground_ids, num_platforms, n)
        transfers[~served] = -1
        np.fill_diagonal(transfers, 0)
        offdiag = ~np.eye(n, dtype=bool)
        d["d_0"] = float(demand[served & (transfers == 0) & offdiag].sum() / total)
        d["d_1"] = float(demand[served & (transfers == 1)].sum() / total)
        d["d_2"] = float(demand[served & (transfers == 2)].sum() / total)
        d["d_3p"] = float(demand[served & (transfers >= 3)].sum() / total)
    d["d_un"] = float(demand[~served].sum() / total)

    return AssignmentResult(travel_time=travel_time, transfers=transfers,
                            C_p=C_p, C_o=C_o, d=d)


# broj presedanja = broj ukrcavanja (prelaz zemlja -> platforma) minus 1;
# hod po predecessor matrici, grafovi su mali pa je python petlja ok
def _count_transfers(pred, ground_ids, num_platforms, n):
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
