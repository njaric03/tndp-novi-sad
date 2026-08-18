from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# standardni transfer penal iz literature
TRANSFER_PENALTY_MIN = 5.0
# nepokriven par placa kao da ide peske: 20/5 (brzina) * 2.0 (tezina peske minuta) = 8
# nije iznad najgoreg opsluzenog para (taj ide do 20.9x), samo iznad 98% traznje - ostatak
# se optimizatoru svejedno isplati ispustiti (checks.py, provera 2)
BUS_SPEED_KMH, WALK_SPEED_KMH = 20.0, 5.0
TEZINA_PESACENJA = 2.0
UNSERVED_FACTOR = BUS_SPEED_KMH / WALK_SPEED_KMH * TEZINA_PESACENJA


@dataclass
class AssignmentResult:
    travel_time: np.ndarray  # (n, n), inf za nepokrivene parove
    transfers: np.ndarray    # (n, n), -1 za nepokrivene; None ako se ne racuna
    C_p: float               # prosecno vreme po putniku, samo opsluzeni parovi
    C_p_all: float           # isto, ali nepokriveni naplaceni po UNSERVED_FACTOR
    C_o: float               # ukupno vreme voznje linija u jednom smeru
    d: dict                  # d_0, d_1, d_2, d_3p, d_un udeli demanda
    boardings: np.ndarray = None  # (R,) ulazaka po liniji; None ako se ne racuna
    max_load: np.ndarray = None   # (R,) najopterecenija deonica linije

    @property
    def is_connected(self):
        return self.d["d_un"] == 0.0


# skale za normalizaciju putnickog i operaterskog clana; obe zavise samo od
# grada, ne od R i max_len, pa jedan grad ima jedne skale za sve konfiguracije
def cost_scales(city):
    return city.street_shortest_mean_demand, city.mst_time


# jedina funkcija cilja: oba clana su odnos prema svojoj donjoj granici
def objective(result, scales, alpha=0.5):
    cp_scale, co_scale = scales
    return alpha * result.C_p_all / cp_scale + (1 - alpha) * result.C_o / co_scale


# isti cilj, ali od gotove mreze: presedanja se ne racunaju jer u cilj ne ulaze
def network_objective(city, network, scales, alpha=0.5):
    return objective(assign(city, network, compute_transfers=False), scales, alpha)


# passenger assignment preko "route grafa": platforma = (linija, pozicija), plus zemaljski cvor po cvoru grada
def assign(city, network, compute_transfers=True, compute_loads=False, headways=None):
    n = city.n
    routes = network.routes
    offsets = np.cumsum([0] + [len(r) for r in routes])
    num_platforms = int(offsets[-1])
    ground = num_platforms  # zemaljski cvor grada i ima indeks ground + i

    # ulazak kosta ili fiksni penal iz literature ili, kad su frekvencije poznate
    board = ([TRANSFER_PENALTY_MIN] * len(routes) if headways is None
             else [float(h) / 2.0 for h in headways])

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
            weights += [board[ri], 0.0]

    size = num_platforms + n
    graph = csr_matrix((np.array(weights), (np.array(rows), np.array(cols))),
                       shape=(size, size))

    ground_ids = np.arange(ground, ground + n)
    trace = compute_transfers or compute_loads
    if trace:
        dist, pred = dijkstra(graph, directed=True, indices=ground_ids,
                              return_predecessors=True)
    else:
        dist = dijkstra(graph, directed=True, indices=ground_ids)

    # sa fiksnim penalom put sa k presedanja plati (k+1) penala, pa se prvi skida, penal je tu proxy za neugodnost presedanja
    travel_time = dist[:, ground_ids]
    if headways is None:
        travel_time = travel_time - TRANSFER_PENALTY_MIN
    np.fill_diagonal(travel_time, 0.0)

    demand = city.demand
    total = demand.sum()
    served = np.isfinite(travel_time)
    served_demand = demand[served].sum()
    C_p = float((demand[served] * travel_time[served]).sum() / served_demand) \
        if served_demand > 0 else float("inf")
    # C_p_all: isti prosek nad SVIM parovima, gde nepokriveni placaju UNSERVED_FACTOR puta ulicno najkrace vreme
    charged = np.where(served, travel_time, UNSERVED_FACTOR * city.street_shortest)
    C_p_all = float((demand * charged).sum() / total)
    C_o = float(network.route_times(city).sum())

    transfers, boardings, max_load = None, None, None
    d = {"d_0": 0.0, "d_1": 0.0, "d_2": 0.0, "d_3p": 0.0}
    if trace:
        transfers, boardings, max_load = _trace(
            pred, ground_ids, num_platforms, n, offsets, demand, routes, compute_loads,
            city=city, headways=headways)
        transfers[~served] = -1
        np.fill_diagonal(transfers, 0)
        offdiag = ~np.eye(n, dtype=bool)
        d["d_0"] = float(demand[served & (transfers == 0) & offdiag].sum() / total)
        d["d_1"] = float(demand[served & (transfers == 1)].sum() / total)
        d["d_2"] = float(demand[served & (transfers == 2)].sum() / total)
        d["d_3p"] = float(demand[served & (transfers >= 3)].sum() / total)
    d["d_un"] = float(demand[~served].sum() / total)

    return AssignmentResult(travel_time=travel_time, transfers=transfers,
                            C_p=C_p, C_p_all=C_p_all, C_o=C_o, d=d,
                            boardings=boardings, max_load=max_load)


# Koliko puta se ista direktna voznja (od_cvora, do_cvora) nudi na razlicitim linijama.
# Sluzi SAMO za podelu opterecenja po linijama: dodela najkracim putem daje ceo par
# jednoj liniji, pa u koridoru sa dve paralelne linije druga dobija tacno nulu, sto
# je artefakt a ne stvarnost. Cilj se ovim ne dira, objective() gleda travel_time i
# route_times, a oni ostaju netaknuti.
#
# Ovo je aproksimacija Spiess-Florianovog modela strategija: deli se samo voznja koju
# je najkraci put vec izabrao, ne trazi se optimalna strategija iznova.
TOL_PARALELNE = 1.25  # alternativa se broji ako nije duza od 1.25x izabrane


def _direct_legs(city, routes):
    veze = {}
    for ri, r in enumerate(routes):
        kum = [0.0]
        for a, b in zip(r, r[1:]):
            kum.append(kum[-1] + float(city.street_time[a, b]))
        for p in range(len(r)):
            for q in range(len(r)):
                if p != q:
                    veze.setdefault((r[p], r[q]), []).append(
                        (ri, p, q, abs(kum[q] - kum[p])))
    return veze


# udeli po liniji za jednu voznju: obrnuto proporcionalno intervalu sledjenja,
# jer putnik ulazi u prvo vozilo koje naidje
def _udeli(kandidati, headways):
    if headways is None:
        w = np.ones(len(kandidati))
    else:
        w = np.array([1.0 / max(float(headways[k[0]]), 1e-9) for k in kandidati])
    s = w.sum()
    return w / s if s > 0 else np.full(len(kandidati), 1.0 / len(kandidati))


# jedna voznja (linija, ulazna pozicija, izlazna pozicija) razdeljena po paralelnim linijama
def _dodaj_nogu(ri, ulaz, izlaz, w, routes, veze, headways, boardings, seg):
    if w <= 0.0:
        return
    kand = [(ri, ulaz, izlaz, 0.0)]
    if veze is not None and ulaz != izlaz:
        r = routes[ri]
        svi = veze.get((r[ulaz], r[izlaz]), [])
        moje = next((t for rj, p, q, t in svi
                     if rj == ri and p == ulaz and q == izlaz), None)
        if moje is not None:
            # izabrana voznja mora da ostane u skupu i kad je tolerancija podesena nisko
            kand = [x for x in svi if x[3] <= TOL_PARALELNE * moje + 1e-9] or kand
    for (rj, p, q, _), u in zip(kand, _udeli(kand, headways)):
        boardings[rj] += w * u
        for s in range(min(p, q), max(p, q)):
            seg[rj][s] += w * u


# hod unazad po predecessor matrici, od cilja ka izvoru
def _trace(pred, ground_ids, num_platforms, n, offsets, demand, routes, want_loads,
           city=None, headways=None):
    transfers = np.full((n, n), -1, dtype=int)
    if not want_loads:
        boardings = seg = None
    else:
        boardings = np.zeros(len(routes))
        seg = [np.zeros(max(len(r) - 1, 0)) for r in routes]
        # platforma -> (linija, pozicija na liniji)
        plat_route = np.zeros(num_platforms, dtype=int)
        plat_pos = np.zeros(num_platforms, dtype=int)
        for ri, route in enumerate(routes):
            base = int(offsets[ri])
            plat_route[base:base + len(route)] = ri
            plat_pos[base:base + len(route)] = np.arange(len(route))

    veze = _direct_legs(city, routes) if (want_loads and city is not None) else None

    for si in range(n):
        p_row = pred[si]
        for tj in range(n):
            if si == tj:
                continue
            cur = ground_ids[tj]
            nboard = 0
            # voznje se skupljaju kao (linija, ulazna pozicija, izlazna pozicija)
            noge, izlaz = [], None
            while True:
                prev = p_row[cur]
                if prev < 0:
                    break
                if want_loads and prev < num_platforms and cur < num_platforms:
                    # hod je unazad, pa je cur bliži izlazu: izlazna pozicija je prva vidjena
                    if izlaz is None:
                        izlaz = plat_pos[cur]
                if prev >= num_platforms and cur < num_platforms:
                    nboard += 1
                    if want_loads:
                        ulaz = plat_pos[cur]
                        noge.append((plat_route[cur], ulaz,
                                     izlaz if izlaz is not None else ulaz))
                        izlaz = None
                cur = prev
            if cur == ground_ids[si] and nboard > 0:
                transfers[si, tj] = nboard - 1
                if want_loads:
                    w = demand[si, tj]
                    for ri, ulaz, izl in noge:
                        _dodaj_nogu(ri, ulaz, izl, w, routes, veze, headways,
                                    boardings, seg)

    max_load = None
    if want_loads:
        max_load = np.array([s.max() if s.size else 0.0 for s in seg])
    return transfers, boardings, max_load
