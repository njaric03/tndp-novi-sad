from pathlib import Path

import pytest

from tndp.baselines.greedy import greedy_network
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign
from tndp.core.io import load_benchmark_city

DATA = Path(__file__).parent.parent / "data" / "benchmarks" / "Mandl" / "Mandl1"


@pytest.fixture(scope="module")
def mandl():
    return load_benchmark_city(DATA)


def test_random_search(mandl):
    net, obj = random_search(mandl, num_routes=6, min_len=2, max_len=8,
                             num_samples=200, seed=0)
    assert net.check(mandl, num_routes=6, min_len=2, max_len=8) == []
    assert assign(mandl, net).is_connected
    # isti seed mora dati isti rezultat
    _, obj2 = random_search(mandl, num_routes=6, min_len=2, max_len=8,
                            num_samples=200, seed=0)
    assert obj == obj2


def test_greedy(mandl):
    net, obj = greedy_network(mandl, 6, 2, 8, alpha=0.5)
    assert net.check(mandl, num_routes=6, min_len=2, max_len=8) == []
    res = assign(mandl, net)
    assert res.is_connected
    # literatura za Mandl daje C_p izmedju ~10.1 i ~15.1, greedy mora upasti
    # u razuman opseg i tuci random search
    assert 10.0 <= res.C_p <= 16.0
    _, rand_obj = random_search(mandl, 6, 2, 8, num_samples=200, seed=0)
    assert obj <= rand_obj
