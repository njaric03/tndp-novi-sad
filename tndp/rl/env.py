import numpy as np

from tndp.core.assignment import assign, cost_scales, objective
from tndp.core.network import TransitNetwork, is_duplicate

# MDP po uzoru na Holliday: epizoda gradi svih R linija redom. za svaku
# liniju prvo se bira početni čvor, pa se naizmenično bira proširenje ili
# halt kad je dužina u [min_len, max_len]. nevalidni potezi se maskiraju.
#
# Akcija proširenja je PAR (kraj, čvor), ne samo čvor. Ranije je akcija bila
# indeks čvora, a strana se pogađala pravilom "ako je sused repa, dopiši na
# rep, inače na početak" — pa je za čvor susedan OBA kraja varijanta "na
# početak" bila nedostižna. Na Delaunay grafovima je takvih poteza ~13%,
# dakle politika nije mogla ni da vidi ni da nauči taj deo prostora akcija.

EXTEND, HALT = 0, 1   # tip odluke koju policy trenutno donosi
HEAD, TAIL = 0, 1     # na koji kraj linije se čvor dodaje


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
        self.stuck = 0  # koliko puta je linija zatvorena kraća od min_len
        return self

    @property
    def done(self):
        return len(self.routes) == self.num_routes

    # čvorovi na krajevima tekuće linije, ili None dok je linija prazna
    @property
    def ends(self):
        return (self.current[0], self.current[-1]) if self.current else None

    # snapshot i vraćanje stanja, za pretragu stabla (MCTS)
    def clone_state(self):
        return ([r[:] for r in self.routes], self.current[:], self.stuck)

    def set_state(self, state):
        routes, current, stuck = state
        self.routes = [r[:] for r in routes]
        self.current = current[:]
        self.stuck = stuck

    # koju odluku policy sada donosi i koji su potezi dozvoljeni.
    # maska je oblika (2, n): mask[HEAD, v] = "dodaj v na početak linije".
    # kod HALT odluke akcija -1 (završi liniju) je uvek dozvoljena, a
    # potezi iz maske znače "ipak produži".
    def decision(self):
        n = self.city.n
        mask = np.zeros((2, n), dtype=bool)

        if not self.current:
            # start linije: bilo koji dozvoljen terminal. strana je nebitna,
            # pa se koristi samo HEAD polovina da akcija ostane jednoznačna.
            mask[HEAD] = self.city.terminal
            return EXTEND, mask

        if len(self.current) < self.max_len:
            sides = ((TAIL, self.current[-1]),) if len(self.current) == 1 else \
                    ((HEAD, self.current[0]), (TAIL, self.current[-1]))
            # kod linije od jednog čvora su oba kraja isti čvor, pa bi HEAD i
            # TAIL davali istu liniju (jedna je obrnuta druga) — nudi se samo TAIL
            for side, end in sides:
                for c in self.neighbors[end]:
                    if c not in self.current and not self._dead_end(side, c):
                        mask[side, c] = True

        # halt je zabranjen ako bi linija bila duplikat neke već sagrađene.
        # greedy dekodiranje je deterministično, pa bez ovoga ume da vrati
        # istu liniju više puta — degenerisana mreža koja ipak prolazi kao
        # "R linija".
        can_halt = (len(self.current) >= self.min_len
                    and self.city.terminal[self.current[0]]
                    and self.city.terminal[self.current[-1]]
                    and not is_duplicate(self.current, self.routes))
        # zaglavljeno: nema proširenja a kraj nije dozvoljen. linija se svejedno
        # zatvara, ali se broji kao prekršaj (vidi step) umesto da tiho prođe —
        # hvata je i TransitNetwork.check() u eksperimentima.
        if can_halt or not mask.any():
            return HALT, mask
        return EXTEND, mask

    # Da li potez vodi u stanje iz kog se ne može izaći ispravno: linija bi
    # bila duplikat već sagrađene, a nema se čime dalje produžiti. Halt je
    # tada jedini potez i duplikat izlazi napolje uprkos maski — tako je
    # greedy dekod na synth-gravity-20007 vratio istu liniju dvaput.
    # Provera je jedan korak unapred, dovoljno jer je duplikat moguć tek na
    # punoj dužini linije ili u čvoru bez slobodnih suseda.
    def _dead_end(self, side, node):
        nxt = ([node] + self.current) if side == HEAD else (self.current + [node])
        if not is_duplicate(nxt, self.routes):
            return False
        if len(nxt) >= self.max_len:
            return True
        return not any(c not in nxt for end in (nxt[0], nxt[-1])
                       for c in self.neighbors[end])

    # Poslednja odbrana od duplikata. Maska iz _dead_end pokriva skoro sve, ali
    # ne sve: linija može postati duplikat na dužini manjoj od max_len, pa da
    # joj i svako proširenje bude dead end — tada je halt jedini potez i
    # duplikat izlazi napolje. Tada se linija skraćuje sa repa dok ne prestane
    # da bude duplikat, uz očuvanje min_len i dozvoljenih terminala.
    # Ovo je projekcija akcije na dopustiv skup, ne izbor politike, pa se
    # broji u self.stuck.
    def _dedup(self, route):
        if not is_duplicate(route, self.routes):
            return route
        cut = route[:]
        while len(cut) > self.min_len:
            cut = cut[:-1]
            if not is_duplicate(cut, self.routes) and self.city.terminal[cut[-1]]:
                return cut
        return route  # nije uspelo; check() će ovo prijaviti

    # akcija: ravan indeks u masku (side * n + node), ili -1 za kraj linije
    def step(self, action):
        if action == -1:
            # brojanje ide ovde a ne u decision(), jer decision() nad istim
            # stanjem zovu i MCTS i rollout po više puta — kao sporedni efekat
            # čitanja stanja brojač bi bio naduvan i besmislen
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

    # terminalna nagrada: negativan cilj. cilj je alpha-ponderisan zbir
    # putničkog i operaterskog člana, oba kao odnos prema svojoj donjoj
    # granici; nepokrivena tražnja je već u putničkom članu (C_p_all), pa
    # nema zasebne kazne ni magične konstante.
    def reward(self):
        net = TransitNetwork(routes=self.routes)
        res = assign(self.city, net, compute_transfers=False)
        return -objective(res, self.scales, self.alpha), res
