import numpy as np
from scipy.sparse.csgraph import dijkstra

from tndp.core.assignment import assign, combined_cost
from tndp.core.network import TransitNetwork

# MDP po uzoru na Holliday: epizoda gradi svih R linija redom. za svaku
# liniju prvo se bira početni čvor, pa se naizmenično bira proširenje
# (sused bilo kog kraja linije koji nije već u njoj) ili halt kad je
# dužina u [min_len, max_len]. nevalidni potezi se maskiraju.

EXTEND, HALT = 0, 1  # tip odluke koju policy trenutno donosi


class TndpEnv:
    def __init__(self, city, num_routes, min_len=2, max_len=8, alpha=0.5):
        self.city = city
        self.num_routes = num_routes
        self.min_len, self.max_len = min_len, max_len
        self.alpha = alpha
        n = city.n
        self.neighbors = [np.flatnonzero(np.isfinite(city.street_time[i])
                                         & (np.arange(n) != i)) for i in range(n)]
        # skale za normalizaciju nagrade: donja granica C_p je demand-ponderisano
        # najkraće vreme ulicom, skala C_o je gruba dužina cele mreže
        street = np.where(np.isfinite(city.street_time), city.street_time, 0.0)
        sp = dijkstra(street, directed=False)
        self.cp_scale = float((city.demand * sp).sum() / city.demand.sum())
        finite = np.isfinite(city.street_time) & (city.street_time > 0)
        self.co_scale = num_routes * (max_len - 1) * float(city.street_time[finite].mean())
        self.reset()

    def reset(self):
        self.routes = []
        self.current = []
        return self

    @property
    def done(self):
        return len(self.routes) == self.num_routes

    # koju odluku policy sada donosi i koji su čvorovi dozvoljeni.
    # kod HALT odluke akcija -1 (završi liniju) je uvek dozvoljena,
    # a čvorovi iz maske znače "ipak produži"
    def decision(self):
        if not self.current:
            return EXTEND, np.ones(self.city.n, dtype=bool)  # start: bilo koji čvor
        mask = np.zeros(self.city.n, dtype=bool)
        if len(self.current) < self.max_len:
            for end in (self.current[0], self.current[-1]):
                for c in self.neighbors[end]:
                    if c not in self.current:
                        mask[c] = True
        if len(self.current) >= self.min_len or not mask.any():
            return HALT, mask
        return EXTEND, mask

    # akcija: indeks čvora, ili -1 za kraj linije
    def step(self, action):
        if action == -1:
            self.routes.append(self.current)
            self.current = []
            return
        node = int(action)
        if not self.current:
            self.current = [node]
        elif node in self.neighbors[self.current[-1]]:
            self.current.append(node)
        else:
            self.current.insert(0, node)

    # terminalna nagrada: negativan normalizovan cost, plus kazna za
    # nepokriven demand (maskiranje ne može da garantuje povezanost)
    def reward(self):
        net = TransitNetwork(routes=self.routes)
        res = assign(self.city, net, compute_transfers=False)
        cost = self.alpha * res.C_p / self.cp_scale \
            + (1 - self.alpha) * res.C_o / self.co_scale
        if not np.isfinite(cost):  # ništa pokriveno
            cost = 10.0
        return -(cost + 5.0 * res.d["d_un"]), res
