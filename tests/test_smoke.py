# Jedan sazet smoke test umesto ranija tri fajla (generator, baselines, RL).
# Nije "pokrivenost" nego zastita od regresije u delovima koji su najskuplji
# za rucnu proveru: prostor akcija MDP-a, oblik logita i to da svaka metoda
# vraca mrezu koja postuje ogranicenja.

import numpy as np
import pytest

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign, cost_scales, objective
from tndp.core.network import TransitNetwork
from tndp.rl.env import EXTEND, HALT, HEAD, TAIL, TndpEnv
from tndp.synth.generator import generate_city

R, LO, HI = 3, 2, 6


@pytest.fixture(scope="module")
def city():
    return generate_city(n=18, seed=0, demand_mode="gravity")


def test_generator_valid_and_deterministic(city):
    assert city.validate() == []
    assert np.array_equal(generate_city(n=18, seed=0).demand, city.demand)
    # gravity mora biti koncentrisaniji od uniformnog
    def top_share(c):
        v = np.sort(c.demand[np.triu_indices(c.n, 1)])[::-1]
        return v[:max(1, len(v) // 10)].sum() / v.sum()
    assert top_share(city) > top_share(generate_city(n=18, seed=0,
                                                     demand_mode="uniform")) + 0.1


def test_baselines_produce_valid_networks(city):
    scales = cost_scales(city)
    nets = {
        "random": random_search(city, R, LO, HI, num_samples=50, alpha=0.5)[0],
        "greedy": greedy_network(city, R, LO, HI, alpha=0.5)[0],
        "hill_climb": hill_climb(city, R, LO, HI, alpha=0.5, max_evals=200)[0],
    }
    for name, net in nets.items():
        assert net.check(city, R, LO, HI) == [], name
    # hill climbing krece od greedy resenja pa ne sme da bude gori od njega
    obj = {n: objective(assign(city, v, compute_transfers=False), scales, 0.5)
           for n, v in nets.items()}
    assert obj["hill_climb"] <= obj["greedy"] + 1e-9
    assert obj["greedy"] < obj["random"]


def test_duplicate_routes_are_rejected(city):
    nb = int(city.neighbors[0][0])
    assert any("duplirane" in p for p in
               TransitNetwork([[0, nb]] * R).check(city, R, LO, HI))


# akcija je par (kraj, cvor): za cvor susedan OBA kraja obe varijante moraju
# biti dostupne i davati razlicite linije. ranije je "na pocetak" bilo
# nedostizno jer se strana pogadjala iz susedstva.
def test_action_space_is_unambiguous(city):
    env = TndpEnv(city, R, LO, HI)
    decision, mask = env.decision()
    assert decision == EXTEND and mask[HEAD].all() and not mask[TAIL].any()
    env.step(HEAD * city.n + 3)
    _, mask = env.decision()
    assert not mask[HEAD].any()  # linija od jednog cvora nudi samo TAIL
    env.step(TAIL * city.n + int(np.flatnonzero(mask[TAIL])[0]))

    _, mask = env.decision()
    both = set(np.flatnonzero(mask[HEAD])) & set(np.flatnonzero(mask[TAIL]))
    if both:
        v, state = int(next(iter(both))), env.clone_state()
        outs = []
        for side in (HEAD, TAIL):
            env.set_state(state)
            env.step(side * city.n + v)
            outs.append(env.current[:])
        assert outs[0] != outs[1] and outs[0][0] == v and outs[1][-1] == v
        env.set_state(state)


def test_random_episode_is_valid(city):
    rng = np.random.default_rng(0)
    env = TndpEnv(city, R, LO, HI)
    while not env.done:
        decision, mask = env.decision()
        opts = list(np.flatnonzero(mask.reshape(-1)))
        if decision == HALT:
            opts.append(-1)
        env.step(int(opts[rng.integers(len(opts))]))
    assert env.stuck == 0
    assert TransitNetwork(env.routes).check(city, R, LO, HI) == []
    assert np.isfinite(env.reward()[0])


def test_policy_and_decoders(city):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from tndp.rl.evaluate import decode, decode_sampling, rollout
    from tndp.rl.mcts import mcts_decode
    from tndp.rl.model import TndpPolicy

    torch.manual_seed(0)
    policy = TndpPolicy()
    env = TndpEnv(city, R, LO, HI)
    net, reward, res, logp, ent = rollout(policy, env, sample=True,
                                          gen=torch.Generator().manual_seed(0))
    assert net.check(city, R, LO, HI) == []
    (-reward * logp - 0.01 * ent).backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    for dec in (lambda: decode(policy, city, R, LO, HI),
                lambda: decode_sampling(policy, city, R, k=4, min_len=LO, max_len=HI),
                lambda: mcts_decode(policy, city, R, LO, HI, sims=8)):
        assert dec()[0].check(city, R, LO, HI) == []
    # uzorkovanje ne sme da dira globalni RNG
    assert torch.initial_seed() == 0
