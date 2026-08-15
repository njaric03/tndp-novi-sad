# Ulaz u mrezu: sta grad i stanje epizode postaju pre nego sto ih GATv2 vidi

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.stats import norm, rankdata


def rank_normal(v):
    return norm.ppf(rankdata(v) / (len(v) + 1.0))


# Skup featurea se zadaje kao lista dodataka na osnovnih 13
EXTRAS = ("betweenness", "coreness", "closeness")   # svaki dodaje jednu kolonu
ALL_FEATURES = ("rank-degree",) + EXTRAS
ALIASES = {"v1": (), "v2": ALL_FEATURES}
BASE_COUNT = 13


def spec(features):
    if isinstance(features, str):
        features = ALIASES[features]
    wanted = set(features)
    unknown = wanted - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"nepoznat feature: {sorted(unknown)}; dozvoljeni {ALL_FEATURES}")
    return tuple(f for f in ALL_FEATURES if f in wanted)


def num_features(features):
    return BASE_COUNT + sum(1 for f in spec(features) if f in EXTRAS)


# Udeo traznje koja PROLAZI kroz cvor iduci najkracim ulicnim putem, ne racunajuci polazak i dolazak
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


# k-core dekompozicija ljustenjem cvora najmanjeg stepena (Batagelj-Zaversnik)
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


# mere iz analize kompleksnih mreza, sve kroz istu rang transformaciju kao i ostali featuri
def _network_features(city, extras):
    if city._netfeat is None:
        city._netfeat = {}
    for name in extras:
        if name in city._netfeat:
            continue
        if name == "betweenness":
            v = _betweenness(city)
        elif name == "coreness":
            v = _coreness(city)
        else:
            v = (city.n - 1) / np.maximum(city.street_shortest.sum(1), 1e-9)
        city._netfeat[name] = rank_normal(v)
    return np.column_stack([city._netfeat[i] for i in extras])


# Deo feature-a ne zavisi od stanja epizode nego samo od grada
def _static_node_features(city, extras):
    key = "rank-degree" in extras
    if city._feat is None:
        city._feat = {}
    if key not in city._feat:
        n = city.n
        coords = (city.coords - city.coords.mean(0)) / (city.coords.std(0) + 1e-6)
        # v1 deli stepen proizvoljnom konstantom 4.0; rang transformacija resava isto sto i kod traznje
        degree_raw = [len(nb) for nb in city.neighbors]
        degree = rank_normal(degree_raw) if key else np.array(degree_raw) / 4.0
        # koliko je traznja koncentrisana: udeo u top 10% parova
        vals = np.sort(city.demand[np.triu_indices(n, 1)])[::-1]
        conc = float(vals[:max(1, len(vals) // 10)].sum() / vals.sum())
        city._feat[key] = np.column_stack([
            coords,
            rank_normal(city.demand.sum(1)),
            rank_normal(city.demand.sum(0)),
            degree,
            np.full(n, conc),
        ])
    return city._feat[key]


def node_features(env, features="v1"):
    city = env.city
    n = city.n
    extras = spec(features)
    static = _static_node_features(city, extras)
    measures = [d for d in extras if d in EXTRAS]
    if measures:
        static = np.column_stack([static, _network_features(city, measures)])
    covered = np.zeros(n)
    for r in env.routes:
        covered[r] = 1.0
    in_current = np.zeros(n)
    # pocetak i rep razdvojeni, jer je akcija par (kraj, cvor), politika mora da razlikuje na koji kraj kaci
    is_head = np.zeros(n)
    is_tail = np.zeros(n)
    if env.current:
        in_current[env.current] = 1.0
        is_head[env.current[0]] = 1.0
        is_tail[env.current[-1]] = 1.0
    progress = len(env.routes) / env.num_routes
    # duzina tekuce linije u odnosu na max_len: bez toga politika ne vidi koliko joj je prostora ostalo do halt-a
    fill = len(env.current) / env.max_len
    x = np.column_stack([static, covered, in_current, is_head, is_tail,
                         np.full(n, progress), np.full(n, fill),
                         np.full(n, env.alpha)])
    return torch.tensor(x, dtype=torch.float32)


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
