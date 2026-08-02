import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm, rankdata
from torch_geometric.nn import GATv2Conv

from tndp.rl.env import HALT


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


# Deo feature-a ne zavisi od stanja epizode nego samo od grada. Računa se
# jednom i kešira — ranije se sve ovo (uključujući sortiranja) računalo na
# svakom potezu, a poteza ima ~25 po epizodi.
def _static_node_features(city):
    if city._feat is None:
        n = city.n
        coords = (city.coords - city.coords.mean(0)) / (city.coords.std(0) + 1e-6)
        degree = np.array([len(nb) for nb in city.neighbors]) / 4.0
        # koliko je tražnja koncentrisana: udeo u top 10% parova. rang
        # transformacija briše ovu informaciju iz dem_out/dem_in, pa se
        # vraća kao jedan skalar po gradu
        vals = np.sort(city.demand[np.triu_indices(n, 1)])[::-1]
        conc = float(vals[:max(1, len(vals) // 10)].sum() / vals.sum())
        city._feat = np.column_stack([
            coords,
            rank_normal(city.demand.sum(1)),
            rank_normal(city.demand.sum(0)),
            degree,
            np.full(n, conc),
        ])
    return city._feat


# feature vektor po čvoru za trenutno stanje epizode
def node_features(env):
    city = env.city
    n = city.n
    static = _static_node_features(city)
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


# GATv2 encoder + pointer glava za izbor čvora + halt glava + value glava
class TndpPolicy(nn.Module):
    def __init__(self, in_dim=13, hidden=64, layers=3, heads=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=2)
            for _ in range(layers)])
        # ulaz: [embedding čvora, globalni kontekst, embedding kraja na koji
        # se kači] — poslednji deo je ono što razlikuje "dodaj na početak"
        # od "dodaj na rep"
        self.node_score = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.halt_score = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x, edge_index, edge_attr):
        h = self.embed(x)
        for conv in self.convs:
            h = h + torch.relu(conv(h, edge_index, edge_attr))
        return h

    # logiti nad akcijama: 2n parova (kraj, čvor) po rasporedu maske
    # [HEAD nad svim čvorovima, TAIL nad svim čvorovima], plus halt kao
    # poslednji logit kad je odluka HALT. ends = (head, tail) ili None dok
    # je linija prazna.
    def action_logits(self, h, decision, mask, ends):
        n = h.shape[0]
        context = h.mean(0)
        ctx = context.expand(n, -1)
        if ends is None:  # start linije: nema kraja, kotva je globalni kontekst
            anchors = (ctx, ctx)
        else:
            anchors = (h[ends[0]].expand(n, -1), h[ends[1]].expand(n, -1))
        scores = torch.cat([
            self.node_score(torch.cat([h, ctx, a], dim=1)).squeeze(-1)
            for a in anchors])
        scores = scores.masked_fill(
            ~torch.as_tensor(mask, dtype=torch.bool).reshape(-1), -torch.inf)
        if decision == HALT:
            halt = self.halt_score(context).reshape(1)
            scores = torch.cat([scores, halt])  # poslednji logit = završi liniju
        return scores

    def state_value(self, h):
        return self.value(h.mean(0)).squeeze()
