# Ulaz u mrežu: šta grad i stanje epizode postaju pre nego što ih GATv2 vidi

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.stats import norm, rankdata


# Rang -> približno N(0,1)
def rank_normal(v):
    return norm.ppf(rankdata(v) / (len(v) + 1.0))


# Skup featurea se zadaje kao lista dodataka na osnovnih 13
DODACI = ("betweenness", "coreness", "closeness")   # svaki dodaje jednu kolonu
SVI = ("rank-degree",) + DODACI
ALIJASI = {"v1": (), "v2": SVI}
OSNOVNIH = 13


# lista dodataka u kanonskom redosledu, iz stringa ili iz liste
def spec(features):
    if isinstance(features, str):
        features = ALIJASI[features]
    trazeno = set(features)
    nepoznato = trazeno - set(SVI)
    if nepoznato:
        raise ValueError(f"nepoznat feature: {sorted(nepoznato)}; dozvoljeni {SVI}")
    return tuple(f for f in SVI if f in trazeno)


def num_features(features):
    return OSNOVNIH + sum(1 for f in spec(features) if f in DODACI)


# Udeo tražnje koja PROLAZI kroz čvor idući najkraćim uličnim putem, ne računajući polazak i dolazak
def _betweenness(city):
    dist = np.where(np.isfinite(city.street_time), city.street_time, 0.0)
    _, pred = dijkstra(dist, directed=False, return_predecessors=True)
    through = np.zeros(city.n)
    for s in range(city.n):
        for t in range(city.n):
            w = city.demand[s, t]
            if s == t or w == 0.0:
                continue
            c = pred[s, t]
            while c != s and c >= 0:
                through[c] += w
                c = pred[s, c]
    return through


# k-core dekompozicija ljuštenjem čvora najmanjeg stepena (Batagelj-Zaveršnik)
def _coreness(city):
    n = city.n
    degree = np.array([len(nb) for nb in city.neighbors], dtype=float)
    alive = np.ones(n, dtype=bool)
    coreness = np.zeros(n)
    k = 0.0
    for _ in range(n):
        i = int(np.argmin(np.where(alive, degree, np.inf)))
        k = max(k, degree[i])
        coreness[i] = k
        alive[i] = False
        for j in city.neighbors[i]:
            if alive[j]:
                degree[j] -= 1.0
    return coreness


# mere iz analize kompleksnih mreža, sve kroz istu rang transformaciju kao i ostali featuri
def _network_features(city, dodaci):
    if city._netfeat is None:
        city._netfeat = {}
    for ime in dodaci:
        if ime in city._netfeat:
            continue
        if ime == "betweenness":
            v = _betweenness(city)
        elif ime == "coreness":
            v = _coreness(city)
        else:
            v = (city.n - 1) / np.maximum(city.street_shortest.sum(1), 1e-9)
        city._netfeat[ime] = rank_normal(v)
    return np.column_stack([city._netfeat[i] for i in dodaci])


# Deo feature-a ne zavisi od stanja epizode nego samo od grada
def _static_node_features(city, dodaci):
    kljuc = "rank-degree" in dodaci
    if city._feat is None:
        city._feat = {}
    if kljuc not in city._feat:
        n = city.n
        coords = (city.coords - city.coords.mean(0)) / (city.coords.std(0) + 1e-6)
        # v1 deli stepen konstantom 4.0, a to je ista greška koju rang transformacija rešava kod tražnje: Delaunay ima prosečan
        degree_raw = [len(nb) for nb in city.neighbors]
        degree = rank_normal(degree_raw) if kljuc else np.array(degree_raw) / 4.0
        # koliko je tražnja koncentrisana: udeo u top 10% parova
        vals = np.sort(city.demand[np.triu_indices(n, 1)])[::-1]
        conc = float(vals[:max(1, len(vals) // 10)].sum() / vals.sum())
        city._feat[kljuc] = np.column_stack([
            coords,
            rank_normal(city.demand.sum(1)),
            rank_normal(city.demand.sum(0)),
            degree,
            np.full(n, conc),
        ])
    return city._feat[kljuc]


# feature vektor po čvoru za trenutno stanje epizode
def node_features(env, features="v1"):
    city = env.city
    n = city.n
    dodaci = spec(features)
    static = _static_node_features(city, dodaci)
    mere = [d for d in dodaci if d in DODACI]
    if mere:
        static = np.column_stack([static, _network_features(city, mere)])
    covered = np.zeros(n)
    for r in env.routes:
        covered[r] = 1.0
    in_current = np.zeros(n)
    # početak i rep razdvojeni, jer je akcija par (kraj, čvor), politika mora da razlikuje na koji kraj kači
    is_head = np.zeros(n)
    is_tail = np.zeros(n)
    if env.current:
        in_current[env.current] = 1.0
        is_head[env.current[0]] = 1.0
        is_tail[env.current[-1]] = 1.0
    progress = len(env.routes) / env.num_routes
    # dužina tekuće linije u odnosu na max_len: bez toga politika ne vidi koliko joj je prostora ostalo do halt-a
    fill = len(env.current) / env.max_len
    x = np.column_stack([static, covered, in_current, is_head, is_tail,
                         np.full(n, progress), np.full(n, fill),
                         np.full(n, env.alpha)])
    return torch.tensor(x, dtype=torch.float32)


# ulične ivice u oba smera + tau i demand para kao edge feature
def edge_tensors(city):
    if city._edge is None:
        e = city.street_edges
        idx = np.concatenate([e, e[:, ::-1]]).T
        tau = city.street_time[idx[0], idx[1]]
        dem = city.demand[idx[0], idx[1]]
        attr = np.column_stack([tau / tau.mean(), rank_normal(dem)])
        city._edge = (torch.tensor(idx, dtype=torch.long),
                      torch.tensor(attr, dtype=torch.float32))
    return city._edge


