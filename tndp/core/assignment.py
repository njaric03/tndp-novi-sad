from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# standardni transfer penal iz literature
TRANSFER_PENALTY_MIN = 5.0
# Nepokriven par ne kažnjavamo proizvoljnom konstantom nego ga naplaćujemo
# kao da putnik istu razdaljinu pređe pešice. Faktor je zato odnos brzina:
# autobus 20 km/h (BUS_SPEED_KMH u generatoru) prema pešaku 5 km/h.
# Ulazi u C_p_all, dakle u isti putnički član kao i svi ostali parovi — nema
# zasebne kazne koja bi se odvojeno podešavala.
#
# Vrednost je bitna: mereno na greedy rešenjima, opslužen par putuje samo
# ~1.04x duže od uličnog najkraćeg vremena, pa faktor blizu 1 znači da
# "ne opslužiti" košta koliko i "opslužiti" i pokrivenost se uruši.
# Osetljivost je u tools/metodoloske_provere.py i mora ići uz rezultate.
BUS_SPEED_KMH, WALK_SPEED_KMH = 20.0, 5.0
UNSERVED_FACTOR = BUS_SPEED_KMH / WALK_SPEED_KMH


@dataclass
class AssignmentResult:
    travel_time: np.ndarray  # (n, n), inf za nepokrivene parove
    transfers: np.ndarray    # (n, n), -1 za nepokrivene; None ako se ne računa
    C_p: float               # prosečno vreme po putniku, samo opsluženi parovi
    C_p_all: float           # isto, ali nepokriveni naplaćeni po UNSERVED_FACTOR
    C_o: float               # ukupno vreme vožnje linija u jednom smeru
    d: dict                  # d_0, d_1, d_2, d_3p, d_un udeli demanda

    @property
    def is_connected(self):
        return self.d["d_un"] == 0.0


# skale za normalizaciju putničkog i operaterskog člana. obe su DONJE
# GRANICE istog tipa, da alpha zaista balansira:
#   cp_scale — demand-ponderisano najkraće vreme ulicom (mreža ide svuda),
#   co_scale — ukupno vreme minimalnog razapinjućeg stabla (najmanje mreže
#              koliko treba da svaki čvor bude dostupan).
# ranija co skala R*(max_len-1)*mean(tau) je bila procena gornje granice i
# skoro konstanta po gradu, pa je operaterski član imao upola manji uticaj
# nego putnički. mereno preko kandidat-rešenja, odnos rasipanja ta dva člana
# je sa ovim skalama ~1.1:1 umesto ~2.1:1.
# num_routes i max_len se više ne koriste; ostavljeni su u potpisu da
# pozivaoci ne moraju da se menjaju.
def cost_scales(city, num_routes=None, max_len=None):
    return city.street_shortest_mean_demand, city.mst_time


# jedina funkcija cilja: oba člana su odnos prema svojoj donjoj granici, pa
# je vrednost ~1 kad je mreža blizu teorijskog poda. nepokrivena tražnja je
# već uračunata kroz C_p_all, dakle NEMA zasebne kazne ni magične konstante.
# ovo RL maksimizuje (kao negativnu nagradu) i po ovome se porede sve metode.
def objective(result, scales, alpha=0.5):
    cp_scale, co_scale = scales
    return alpha * result.C_p_all / cp_scale + (1 - alpha) * result.C_o / co_scale


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
    # C_p_all: isti prosek nad SVIM parovima, gde nepokriveni plaćaju
    # UNSERVED_FACTOR puta ulično najkraće vreme. bez ovoga C_p svake metode
    # prosečava preko drugog skupa parova (one sa većim d_un ispuštaju baš
    # najduže parove i time sebi lepšaju C_p), pa nije uporediv.
    charged = np.where(served, travel_time, UNSERVED_FACTOR * city.street_shortest)
    C_p_all = float((demand * charged).sum() / total)
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
                            C_p=C_p, C_p_all=C_p_all, C_o=C_o, d=d)


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
