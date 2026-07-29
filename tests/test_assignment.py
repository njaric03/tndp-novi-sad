"""Unit testovi za passenger assignment na rucno izracunatom toy primeru."""

import numpy as np
import pytest

from tndp.core.assignment import assign, combined_cost
from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork


@pytest.fixture
def toy_city():
    """Put 0-1-2 plus ogranak 1-3; sva vremena 10 min.

    Demand: 0->2 (linija A direktno), 0->3 (A pa B, jedno presedanje).
    """
    n = 4
    street = np.full((n, n), np.inf)
    np.fill_diagonal(street, 0.0)
    for a, b in [(0, 1), (1, 2), (1, 3)]:
        street[a, b] = street[b, a] = 10.0
    demand = np.zeros((n, n))
    demand[0, 2] = demand[2, 0] = 100.0
    demand[0, 3] = demand[3, 0] = 50.0
    coords = np.zeros((n, 2))
    return CityGraph(coords=coords, street_time=street, demand=demand, name="toy")


def test_direct_and_transfer_times(toy_city):
    net = TransitNetwork(routes=[[0, 1, 2], [1, 3]])
    res = assign(toy_city, net)
    # 0->2 direktno linijom A: 20 min, bez presedanja
    assert res.travel_time[0, 2] == pytest.approx(20.0)
    assert res.transfers[0, 2] == 0
    # 0->3: A do cvora 1 (10), presedanje (5), B do cvora 3 (10)
    assert res.travel_time[0, 3] == pytest.approx(25.0)
    assert res.transfers[0, 3] == 1
    # C_p: (200 * 20 + 100 * 25) / 300
    assert res.C_p == pytest.approx((200 * 20 + 100 * 25) / 300)
    # C_o: 20 + 10
    assert res.C_o == pytest.approx(30.0)
    assert res.d["d_0"] == pytest.approx(200 / 300)
    assert res.d["d_1"] == pytest.approx(100 / 300)
    assert res.d["d_un"] == 0.0
    assert res.is_connected


def test_unserved_pairs(toy_city):
    net = TransitNetwork(routes=[[0, 1, 2]])
    res = assign(toy_city, net)
    assert np.isinf(res.travel_time[0, 3])
    assert res.transfers[0, 3] == -1
    assert res.d["d_un"] == pytest.approx(100 / 300)
    assert not res.is_connected


def test_combined_cost(toy_city):
    net = TransitNetwork(routes=[[0, 1, 2], [1, 3]])
    res = assign(toy_city, net)
    assert combined_cost(res, alpha=1.0) == pytest.approx(res.C_p)
    assert combined_cost(res, alpha=0.0) == pytest.approx(res.C_o)


def test_fast_path_skips_transfers(toy_city):
    net = TransitNetwork(routes=[[0, 1, 2], [1, 3]])
    fast = assign(toy_city, net, compute_transfers=False)
    full = assign(toy_city, net, compute_transfers=True)
    assert fast.transfers is None
    assert fast.C_p == pytest.approx(full.C_p)
    assert fast.d["d_un"] == full.d["d_un"]


def test_constraints_check(toy_city):
    ok = TransitNetwork(routes=[[0, 1, 2], [1, 3]])
    assert ok.check(toy_city, num_routes=2, min_len=2, max_len=3) == []
    bad = TransitNetwork(routes=[[0, 2], [1, 1], [3]])
    problems = bad.check(toy_city, num_routes=2, min_len=2, max_len=3)
    assert any("ivica" in p for p in problems)      # 0-2 ne postoji
    assert any("ponovljen" in p for p in problems)  # 1-1
    assert any("duzina" in p for p in problems)     # [3] prekratka
    assert any("broj linija" in p for p in problems)
