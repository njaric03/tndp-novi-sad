import numpy as np
import pytest

from tndp.core.assignment import assign
from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork


# put 0-1-2 plus ogranak 1-3, sva vremena 10 min; demand 0->2 ide linijom
# A direktno, 0->3 mora A pa B sa jednim presedanjem
@pytest.fixture
def toy_city():
    n = 4
    street = np.full((n, n), np.inf)
    np.fill_diagonal(street, 0.0)
    for a, b in [(0, 1), (1, 2), (1, 3)]:
        street[a, b] = street[b, a] = 10.0
    demand = np.zeros((n, n))
    demand[0, 2] = demand[2, 0] = 100.0
    demand[0, 3] = demand[3, 0] = 50.0
    return CityGraph(coords=np.zeros((n, 2)), street_time=street, demand=demand, name="toy")


def test_toy_example_by_hand(toy_city):
    net = TransitNetwork(routes=[[0, 1, 2], [1, 3]])
    res = assign(toy_city, net)
    # 0->2 direktno linijom A: 20 min; 0->3: A do cvora 1 (10),
    # presedanje (5), B do cvora 3 (10) = 25 min
    assert res.travel_time[0, 2] == pytest.approx(20.0)
    assert res.transfers[0, 2] == 0
    assert res.travel_time[0, 3] == pytest.approx(25.0)
    assert res.transfers[0, 3] == 1
    assert res.C_p == pytest.approx((200 * 20 + 100 * 25) / 300)
    assert res.C_o == pytest.approx(30.0)
    assert res.d["d_0"] == pytest.approx(200 / 300)
    assert res.d["d_1"] == pytest.approx(100 / 300)
    assert res.is_connected
    # brza varijanta bez rekonstrukcije puteva mora dati isti C_p
    fast = assign(toy_city, net, compute_transfers=False)
    assert fast.transfers is None
    assert fast.C_p == pytest.approx(res.C_p)


def test_unserved_pairs(toy_city):
    # bez linije ka cvoru 3 taj demand ostaje nepokriven
    net = TransitNetwork(routes=[[0, 1, 2]])
    res = assign(toy_city, net)
    assert np.isinf(res.travel_time[0, 3])
    assert res.d["d_un"] == pytest.approx(100 / 300)
    assert not res.is_connected


def test_constraints_check(toy_city):
    ok = TransitNetwork(routes=[[0, 1, 2], [1, 3]])
    assert ok.check(toy_city, num_routes=2, min_len=2, max_len=3) == []
    bad = TransitNetwork(routes=[[0, 2], [1, 1], [3]])
    problems = bad.check(toy_city, num_routes=2, min_len=2, max_len=3)
    assert any("ivica" in p for p in problems)      # 0-2 ne postoji
    assert any("ponovljen" in p for p in problems)  # 1-1
    assert any("dužina" in p for p in problems)     # [3] prekratka
    assert any("broj linija" in p for p in problems)
