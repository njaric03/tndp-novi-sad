import math

import torch

from tndp.core.assignment import assign
from tndp.core.network import TransitNetwork
from tndp.rl.env import HALT, TndpEnv
from tndp.rl.model import edge_tensors, node_features

# MCTS dekodiranje sa naučenim priorima (PUCT), po uzoru na AlphaTransit ali
# bez MCTS-a u treningu. politika daje prior P(a|s) za širenje stabla, a
# vrednost lista se dobija greedy rollout-om iste politike do kraja epizode
# (pouzdanije od value glave koja je trenirana samo na početnom stanju).
# koristi se samo pri evaluaciji.


class _Node:
    __slots__ = ("state", "terminal", "P", "N", "W", "children")

    def __init__(self, state):
        self.state = state
        self.terminal = False
        self.P = {}         # akcija -> prior
        self.N = {}         # akcija -> broj poseta
        self.W = {}         # akcija -> zbir vrednosti
        self.children = {}  # akcija -> _Node


# akcija je indeks čvora, ili -1 za halt
@torch.no_grad()
def _priors(policy, env, edge_index, edge_attr):
    decision, mask = env.decision()
    h = policy.encode(node_features(env), edge_index, edge_attr)
    logits = policy.action_logits(h, decision, mask)
    probs = torch.softmax(logits, dim=0).numpy()
    n = env.city.n
    P = {i: float(probs[i]) for i in range(n) if mask[i]}
    if decision == HALT:
        P[-1] = float(probs[n])  # poslednji logit je halt
    return P


def _make_node(policy, env, edge_index, edge_attr):
    node = _Node(env.clone_state())
    if env.done:
        node.terminal = True
    else:
        node.P = _priors(policy, env, edge_index, edge_attr)
    return node


# vrednost stanja = exp(nagrada) u (0, 1], greedy rollout do kraja.
# greedy je namerno: jedan sampled rollout je previše šumovit i obmane stablo
@torch.no_grad()
def _rollout_value(policy, env, edge_index, edge_attr):
    while not env.done:
        decision, mask = env.decision()
        h = policy.encode(node_features(env), edge_index, edge_attr)
        logits = policy.action_logits(h, decision, mask)
        a = int(logits.argmax())
        is_halt = decision == HALT and a == len(logits) - 1
        env.step(-1 if is_halt else a)
    return math.exp(env.reward()[0])


def _puct(node, c):
    sqrt_total = math.sqrt(sum(node.N.values()) + 1)
    best, best_score = None, -1e18
    for a, p in node.P.items():
        n = node.N.get(a, 0)
        q = node.W[a] / n if n > 0 else 0.0
        score = q + c * p * sqrt_total / (1 + n)
        if score > best_score:
            best, best_score = a, score
    return best


def _step(env, a):
    env.step(-1 if a == -1 else a)


def _simulate(policy, env, root, edge_index, edge_attr, c):
    node, path = root, []
    while True:
        a = _puct(node, c)
        path.append((node, a))
        if a in node.children:
            node = node.children[a]
            if node.terminal:
                env.set_state(node.state)
                value = math.exp(env.reward()[0])
                break
            continue
        # prošireno: napravi dete za akciju a i oceni ga rollout-om
        env.set_state(node.state)
        _step(env, a)
        child = _make_node(policy, env, edge_index, edge_attr)
        node.children[a] = child
        value = math.exp(env.reward()[0]) if child.terminal \
            else _rollout_value(policy, env, edge_index, edge_attr)
        break
    for n, a in path:
        n.N[a] = n.N.get(a, 0) + 1
        n.W[a] = n.W.get(a, 0.0) + value


@torch.no_grad()
def mcts_decode(policy, city, num_routes, min_len=2, max_len=8, alpha=0.5,
                sims=50, c_puct=1.5):
    env = TndpEnv(city, num_routes, min_len, max_len, alpha)
    edge_index, edge_attr = edge_tensors(city)
    env.reset()
    while not env.done:
        root = _make_node(policy, env, edge_index, edge_attr)
        for _ in range(sims):
            _simulate(policy, env, root, edge_index, edge_attr, c_puct)
        best = max(root.N, key=root.N.get)  # najposećenija akcija
        env.set_state(root.state)
        _step(env, best)
    net = TransitNetwork(routes=env.routes)
    return net, assign(city, net)
