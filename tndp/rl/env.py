import numpy as np

from tndp.core.assignment import assign, cost_scales, objective
from tndp.core.network import TransitNetwork, is_duplicate

# MDP po uzoru na Holliday: epizoda gradi svih R linija redom

EXTEND, HALT = 0, 1   # tip odluke koju policy trenutno donosi
HEAD, TAIL = 0, 1     # na koji kraj linije se cvor dodaje


class TndpEnv:
    def __init__(self, city, num_routes, min_len=2, max_len=8, alpha=0.5):
        self.city = city
        self.num_routes = num_routes
        self.min_len, self.max_len = min_len, max_len
        self.alpha = alpha
        self.neighbors = city.neighbors
        self.scales = cost_scales(city)
        self.reset()

    def reset(self):
        self.routes = []
        self.current = []
        self.stuck = 0  # koliko puta je linija zatvorena kraca od min_len
        return self

    @property
    def done(self):
        return len(self.routes) == self.num_routes

    # cvorovi na krajevima tekuce linije, ili None dok je linija prazna
    @property
    def ends(self):
        return (self.current[0], self.current[-1]) if self.current else None

    # snapshot i vracanje stanja, za pretragu stabla (MCTS)
    def clone_state(self):
        return ([r[:] for r in self.routes], self.current[:], self.stuck)

    def set_state(self, state):
        routes, current, stuck = state
        self.routes = [r[:] for r in routes]
        self.current = current[:]
        self.stuck = stuck

    # koju odluku policy sada donosi i koji su potezi dozvoljeni
    def decision(self):
        n = self.city.n
        mask = np.zeros((2, n), dtype=bool)

        if not self.current:
            # start linije: bilo koji dozvoljen terminal
            mask[HEAD] = self.city.terminal
            return EXTEND, mask

        if len(self.current) < self.max_len:
            sides = ((TAIL, self.current[-1]),) if len(self.current) == 1 else \
                    ((HEAD, self.current[0]), (TAIL, self.current[-1]))
            # kod linije od jednog cvora su oba kraja isti cvor, pa bi HEAD i TAIL davali istu liniju (jedna je obrnuta druga)
            for side, end in sides:
                for c in self.neighbors[end]:
                    if c not in self.current and not self._dead_end(side, c):
                        mask[side, c] = True

        # halt je zabranjen ako bi linija bila duplikat neke vec sagradjene
        can_halt = (len(self.current) >= self.min_len
                    and self.city.terminal[self.current[0]]
                    and self.city.terminal[self.current[-1]]
                    and not is_duplicate(self.current, self.routes))
        # zaglavljeno: nema prosirenja a kraj nije dozvoljen
        if can_halt or not mask.any():
            return HALT, mask
        return EXTEND, mask

    # Da li potez vodi u stanje iz kog se ne moze izaci ispravno
    def _dead_end(self, side, node):
        nxt = ([node] + self.current) if side == HEAD else (self.current + [node])
        free = any(c not in nxt for end in (nxt[0], nxt[-1])
                   for c in self.neighbors[end])
        if len(nxt) < self.min_len:
            return not free
        if is_duplicate(nxt, self.routes):
            return len(nxt) >= self.max_len or not free
        return False

    # Poslednja odbrana od duplikata
    def _dedup(self, route):
        if not is_duplicate(route, self.routes):
            return route
        cut = route[:]
        while len(cut) > self.min_len:
            cut = cut[:-1]
            if not is_duplicate(cut, self.routes) and self.city.terminal[cut[-1]]:
                return cut
        return route  # nije uspelo; check() ce ovo prijaviti

    # akcija: ravan indeks u masku (side * n + node), ili -1 za kraj linije
    def step(self, action):
        if action == -1:
            # brojanje ide ovde a ne u decision(), jer decision() nad istim stanjem zovu i MCTS i rollout po vise puta
            route = self._dedup(self.current)
            if len(route) < self.min_len or route is not self.current:
                self.stuck += 1
            self.routes.append(route)
            self.current = []
            return
        side, node = divmod(int(action), self.city.n)
        if not self.current:
            self.current = [node]
        elif side == HEAD:
            self.current.insert(0, node)
        else:
            self.current.append(node)

    # terminalna nagrada: negativan cilj
    def reward(self):
        net = TransitNetwork(routes=self.routes)
        res = assign(self.city, net, compute_transfers=False)
        return -objective(res, self.scales, self.alpha), res
