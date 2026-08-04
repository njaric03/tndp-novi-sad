# Ulaz u mrežu: šta grad i stanje epizode postaju pre nego što ih GATv2 vidi.
# Model je u model.py; ovde je sve što se računa OKO njega.

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.stats import norm, rankdata


# Rang -> približno N(0,1). Tražnja je u gravity režimu po konstrukciji
# lognormalna: sirov udeo `v * n / sum` ima asimetriju ~5.3 na ivicama i
# raspon 40x između medijane i maksimuma, pa nekoliko parova guši sve ostale
# u attention-u. Log to popravlja na sintetici, ali ne i na instancama sa
# nultom tražnjom po parovima (Mandl, Mumford), gde nula postaje izražen
# outlier.
#
# Rang transformacija rešava i drugu, važniju stvar: raspodela feature-a
# postaje **ista bez obzira na instancu** (asimetrija 0.00 i na gravity i na
# uniform i na Mandlu i na Mumfordu, raspon 1.5-2.6). Za model čija je cela
# poenta transfer sa sintetike na stvaran grad, razlika u raspodeli ulaza
# između treninga i testa je ozbiljniji problem od same skale.
#
# Cena je gubitak apsolutnih odnosa ("koliko puta veća tražnja"); to se
# vraća kroz `concentration` feature niže. Sirova tražnja i dalje ulazi u
# funkciju cilja nedirnuta — transformiše se samo ulaz u mrežu.
def rank_normal(v):
    return norm.ppf(rankdata(v) / (len(v) + 1.0))


# Verzije skupa featurea. Stara se zadržava jer su svi dosadašnji modeli na
# njoj trenirani, a checkpoint nosi svoju verziju u cfg-u: evaluirati star
# model novim featurima znači tiho mu promeniti ulaz.
#
#   v1  13 featurea, stepen skaliran konstantom
#   v2  16 featurea: stepen kroz rang transformaciju, plus tri mere iz analize
#       kompleksnih mreža (prolaznost, koreness, bliskost)
NUM_FEATURES = {"v1": 13, "v2": 16}


# Udeo tražnje koja PROLAZI kroz čvor idući najkraćim uličnim putem, ne
# računajući polazak i dolazak — betweenness centralnost ponderisana tražnjom.
# Najbliže što se u jednom broju može reći "ovuda treba da ide magistralna
# linija".
#
# Zašto se ne prepusti mreži da to sama nauči: prolaznost je GLOBALNA veličina,
# a tri GATv2 sloja imaju receptivno polje od tri skoka. Na Mumford3 (n=127)
# čvor nema odakle da sazna gde je u gradu.
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


# k-core dekompozicija ljuštenjem čvora najmanjeg stepena (Batagelj-Zaveršnik).
# Razdvaja gusto jezgro ulične mreže od periferije koja na njega visi; u jezgru
# se linije isplati preklapati, na periferiji ne.
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


# tri mere iz analize kompleksnih mreža, sve kroz istu rang transformaciju kao
# i ostali featuri. Računaju se jednom po gradu i keširaju.
def _network_features(city):
    if city._netfeat is None:
        closeness = (city.n - 1) / np.maximum(city.street_shortest.sum(1), 1e-9)
        city._netfeat = np.column_stack([
            rank_normal(_betweenness(city)),
            rank_normal(_coreness(city)),
            rank_normal(closeness),
        ])
    return city._netfeat


# Deo feature-a ne zavisi od stanja epizode nego samo od grada. Računa se
# jednom i kešira — ranije se sve ovo (uključujući sortiranja) računalo na
# svakom potezu, a poteza ima ~25 po epizodi.
# keš je rečnik po verziji, ne jedna matrica: benchmark skripte puštaju više
# modela preko ISTE liste gradova, pa bi model na v2 featurima inače pokupio
# ono što je keširao model na v1 i to bez ijedne greške
def _static_node_features(city, version="v1"):
    if city._feat is None:
        city._feat = {}
    if version not in city._feat:
        n = city.n
        coords = (city.coords - city.coords.mean(0)) / (city.coords.std(0) + 1e-6)
        # v1 deli stepen konstantom 4.0, a to je ista greška koju rang
        # transformacija rešava kod tražnje: Delaunay ima prosečan stepen ~6,
        # zonski graf Novog Sada 5.3, Mumford instance nešto treće, pa se
        # raspodela ulaza razlikuje između treninga i testa.
        degree_raw = [len(nb) for nb in city.neighbors]
        degree = (np.array(degree_raw) / 4.0 if version == "v1"
                  else rank_normal(degree_raw))
        # koliko je tražnja koncentrisana: udeo u top 10% parova. rang
        # transformacija briše ovu informaciju iz dem_out/dem_in, pa se
        # vraća kao jedan skalar po gradu
        vals = np.sort(city.demand[np.triu_indices(n, 1)])[::-1]
        conc = float(vals[:max(1, len(vals) // 10)].sum() / vals.sum())
        city._feat[version] = np.column_stack([
            coords,
            rank_normal(city.demand.sum(1)),
            rank_normal(city.demand.sum(0)),
            degree,
            np.full(n, conc),
        ])
    return city._feat[version]


# feature vektor po čvoru za trenutno stanje epizode
def node_features(env, version="v1"):
    city = env.city
    n = city.n
    static = _static_node_features(city, version)
    if version != "v1":
        static = np.column_stack([static, _network_features(city)])
    covered = np.zeros(n)
    for r in env.routes:
        covered[r] = 1.0
    in_current = np.zeros(n)
    # početak i rep razdvojeni, jer je akcija par (kraj, čvor) — politika
    # mora da razlikuje na koji kraj kači
    is_head = np.zeros(n)
    is_tail = np.zeros(n)
    if env.current:
        in_current[env.current] = 1.0
        is_head[env.current[0]] = 1.0
        is_tail[env.current[-1]] = 1.0
    progress = len(env.routes) / env.num_routes
    # dužina tekuće linije u odnosu na max_len: bez toga politika ne vidi
    # koliko joj je prostora ostalo do halt-a
    fill = len(env.current) / env.max_len
    x = np.column_stack([static, covered, in_current, is_head, is_tail,
                         np.full(n, progress), np.full(n, fill),
                         np.full(n, env.alpha)])
    return torch.tensor(x, dtype=torch.float32)


# ulične ivice u oba smera + tau i demand para kao edge feature.
# tražnja ide kroz istu rang transformaciju kao i čvorovna (vidi rank_normal);
# tau je blago asimetrično (~0.5) i ostaje kakvo jeste.
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


