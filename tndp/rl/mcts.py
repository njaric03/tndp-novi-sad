import math

import torch

from tndp.core.assignment import assign
from tndp.core.network import TransitNetwork
from tndp.rl.env import HALT, TndpEnv
from tndp.rl.features import edge_tensors, node_features

# MCTS dekodiranje sa naučenim priorima (PUCT), po uzoru na AlphaTransit.
#
# Dve razlike u odnosu na AlphaTransit, obe svesne:
#  - AlphaTransit vrednost lista uzima iz value mreže i MCTS koristi i u
#    treningu, pa value glava uči baš na međustanjima. Kod nas je value glava
#    trenirana samo na početnom stanju, pa bi na međustanjima bila gora od
#    rollout-a; zato je vrednost lista greedy rollout iste politike.
#  - Zbog istog razloga pretraga je samo dekoder, ne ulazi u trening.
# Kod njih se rollout-i izbegavaju jer nagradu daje saobraćajni simulator
# (skupo); kod nas je nagrada jedan Dijkstra na malom grafu, pa rollout-i
# nisu usko grlo.
#
# Vrednosti se normalizuju po stablu (MuZero min-max) umesto ranijeg
# exp(nagrada): sirove vrednosti su bile u uskom opsegu oko 0.26, pa je
# eksploracioni član PUCT-a bio red veličine veći od razlike među akcijama
# i pretraga je faktički uzorkovala iz priora.


class _Node:
    __slots__ = ("state", "terminal", "P", "N", "W", "children")

    def __init__(self, state):
        self.state = state
        self.terminal = False
        self.P = {}         # akcija -> prior
        self.N = {}         # akcija -> broj poseta
        self.W = {}         # akcija -> zbir vrednosti
        self.children = {}  # akcija -> _Node


# min-max normalizacija vrednosti preko celog stabla, da Q i eksploracioni
# član PUCT-a budu na istoj skali bez obzira na apsolutni nivo nagrade
class _Bounds:
    def __init__(self):
        self.lo, self.hi = math.inf, -math.inf

    def update(self, v):
        self.lo, self.hi = min(self.lo, v), max(self.hi, v)

    def norm(self, v):
        if self.hi > self.lo:
            return (v - self.lo) / (self.hi - self.lo)
        return 0.5


# akcija je ravan indeks u masku (side * n + node), ili -1 za halt
@torch.no_grad()
def _priors(policy, env, edge_index, edge_attr):
    decision, mask = env.decision()
    h = policy.encode(node_features(env, policy.version), edge_index, edge_attr)
    logits = policy.action_logits(h, decision, mask, env.ends)
    probs = torch.softmax(logits, dim=0)
    flat = mask.reshape(-1)
    P = {i: float(probs[i]) for i in range(flat.size) if flat[i]}
    if decision == HALT:
        P[-1] = float(probs[-1])  # poslednji logit je halt
    return P


def _make_node(policy, env, edge_index, edge_attr):
    node = _Node(env.clone_state())
    if env.done:
        node.terminal = True
    else:
        node.P = _priors(policy, env, edge_index, edge_attr)
    return node


# vrednost stanja = nagrada greedy rollout-a do kraja. greedy je namerno:
# jedan sampled rollout je previše šumovit i obmane stablo.
@torch.no_grad()
def _rollout_value(policy, env, edge_index, edge_attr):
    while not env.done:
        decision, mask = env.decision()
        h = policy.encode(node_features(env, policy.version), edge_index, edge_attr)
        logits = policy.action_logits(h, decision, mask, env.ends)
        a = int(logits.argmax())
        is_halt = decision == HALT and a == len(logits) - 1
        env.step(-1 if is_halt else a)
    return env.reward()[0]


def _puct(node, c, bounds):
    total = sum(node.N.values())
    sqrt_total = math.sqrt(total + 1)
    # FPU: neposećena akcija nasleđuje tekuću procenu roditelja umesto 0.
    # sa sirovim Q=0 i nagradama oko -1.3 svaka neposećena akcija je
    # izgledala bolje/gore od svih posećenih, zavisno od znaka.
    fpu = bounds.norm(sum(node.W.values()) / total) if total > 0 else 0.5
    best, best_score = None, -1e18
    for a, p in node.P.items():
        n = node.N.get(a, 0)
        q = bounds.norm(node.W[a] / n) if n > 0 else fpu
        score = q + c * p * sqrt_total / (1 + n)
        if score > best_score:
            best, best_score = a, score
    return best


def _simulate(policy, env, root, edge_index, edge_attr, c, bounds):
    node, path = root, []
    while True:
        a = _puct(node, c, bounds)
        path.append((node, a))
        if a in node.children:
            node = node.children[a]
            if node.terminal:
                env.set_state(node.state)
                value = env.reward()[0]
                break
            continue
        # nova akcija: napravi dete i oceni ga rollout-om
        env.set_state(node.state)
        env.step(a)
        child = _make_node(policy, env, edge_index, edge_attr)
        node.children[a] = child
        value = env.reward()[0] if child.terminal \
            else _rollout_value(policy, env, edge_index, edge_attr)
        break
    bounds.update(value)
    for n, a in path:
        n.N[a] = n.N.get(a, 0) + 1
        n.W[a] = n.W.get(a, 0.0) + value


@torch.no_grad()
def mcts_decode(policy, city, num_routes, min_len=2, max_len=8, alpha=0.5,
                sims=50, c_puct=1.5):
    env = TndpEnv(city, num_routes, min_len, max_len, alpha)
    edge_index, edge_attr = edge_tensors(city)
    env.reset()
    bounds = _Bounds()
    root = _make_node(policy, env, edge_index, edge_attr)
    while not env.done:
        for _ in range(sims - sum(root.N.values())):
            _simulate(policy, env, root, edge_index, edge_attr, c_puct, bounds)
        best = max(root.N, key=root.N.get)  # najposećenija akcija
        env.set_state(root.state)
        env.step(best)
        # zadrži podstablo izabrane akcije umesto da se gradi iznova:
        # ranije se ceo posao od ~30 simulacija bacao posle svakog poteza
        root = root.children[best]
    net = TransitNetwork(routes=env.routes)
    return net, assign(city, net)
