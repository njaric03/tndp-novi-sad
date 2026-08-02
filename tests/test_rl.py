import numpy as np
import pytest

from tndp.rl.env import EXTEND, HALT, TndpEnv
from tndp.synth.generator import generate_city


@pytest.fixture(scope="module")
def city():
    return generate_city(n=15, seed=0)


# epizoda sa nasumičnim dozvoljenim akcijama mora dati validnu mrežu
def test_env_random_episode(city):
    rng = np.random.default_rng(0)
    env = TndpEnv(city, num_routes=3, min_len=2, max_len=8)
    while not env.done:
        decision, mask = env.decision()
        options = list(np.flatnonzero(mask))
        if decision == HALT:
            options.append(-1)
        env.step(options[rng.integers(len(options))])
    from tndp.core.network import TransitNetwork
    net = TransitNetwork(routes=env.routes)
    assert net.check(city, num_routes=3, min_len=2, max_len=8) == []
    reward, res = env.reward()
    assert np.isfinite(reward)


def test_env_maskiranje(city):
    env = TndpEnv(city, num_routes=1, min_len=2, max_len=3)
    decision, mask = env.decision()
    assert decision == EXTEND and mask.all()  # start bilo gde
    env.step(0)
    decision, mask = env.decision()
    assert decision == EXTEND  # dužina 1 < min_len, halt još nije opcija
    assert not mask[0]  # ne sme nazad u isti čvor
    env.step(int(np.flatnonzero(mask)[0]))
    decision, mask = env.decision()
    assert decision == HALT  # dužina 2 >= min_len
    env.step(int(np.flatnonzero(mask)[0]))
    _, mask = env.decision()
    assert not mask.any()  # max_len dostignut, samo halt


# jedan gradijentni korak smoke: loss konačan, gradijenti teku
def test_policy_smoke(city):
    torch = pytest.importorskip("torch")
    from tndp.rl.evaluate import rollout
    from tndp.rl.model import TndpPolicy

    torch.manual_seed(0)
    policy = TndpPolicy()
    env = TndpEnv(city, num_routes=2, min_len=2, max_len=8)
    net, reward, res, logp, ent = rollout(policy, env, sample=True)
    assert net.check(city, num_routes=2, min_len=2, max_len=8) == []
    loss = -reward * logp - 0.01 * ent
    loss.backward()
    grads = [p.grad.abs().sum() for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g) for g in grads)


# MCTS dekoder daje validnu mrežu; malo simulacija za brzinu
def test_mcts_decode(city):
    pytest.importorskip("torch")
    from tndp.rl.mcts import mcts_decode
    from tndp.rl.model import TndpPolicy

    policy = TndpPolicy()
    net, res = mcts_decode(policy, city, num_routes=2, min_len=2, max_len=6, sims=8)
    assert net.check(city, num_routes=2, min_len=2, max_len=6) == []
    assert res.C_o > 0
