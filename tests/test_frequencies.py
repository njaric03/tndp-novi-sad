# Frekvencijska faza na ruke: brojevi su izracunati iz definicija u
# core/frequencies.py, ne prepisani iz izlaza koda.

import numpy as np
import pytest

from tndp.core import frequencies as F
from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork


# isti toy grad kao u test_assignment.py: put 0-1-2 plus ogranak 1-3, sve ivice 10 min
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


@pytest.fixture
def toy_net():
    return TransitNetwork(routes=[[0, 1, 2], [1, 3]])


# interval = 60 / (max_load * peak_share / capacity), pa clip na [5, 60]
def test_headways_by_hand():
    # 8000 * 0.10 / 80 = 10 vozila na sat -> 6 min
    assert F.headways([8000.0]) == pytest.approx([6.0])
    # 4800 * 0.10 / 80 = 6 -> 10 min
    assert F.headways([4800.0]) == pytest.approx([10.0])
    # 800 * 0.10 / 80 = 1 -> 60 min, tacno na plafonu
    assert F.headways([800.0]) == pytest.approx([60.0])
    # linija bez ijednog putnika ide na plafon, ne u deljenje nulom
    assert F.headways([0.0]) == pytest.approx([F.H_MAX])
    # 96000 bi trazilo 0.5 min; pod je 5
    assert F.headways([96000.0]) == pytest.approx([F.H_MIN])


def test_headways_respect_bounds_on_random_loads():
    rng = np.random.default_rng(0)
    h = F.headways(rng.uniform(0, 200_000, 500))
    assert np.all(h >= F.H_MIN) and np.all(h <= F.H_MAX)
    assert np.all(np.isfinite(h))


# vozila = ceil(2 * vreme_voznje * (1 + obrt) / interval)
def test_fleet_by_hand(toy_city, toy_net):
    assert toy_net.route_times(toy_city) == pytest.approx([20.0, 10.0])
    # linija A: 2 * 20 * 1.1 = 44 min obilaska, na 10 min -> ceil(4.4) = 5
    # linija B: 2 * 10 * 1.1 = 22 min obilaska, na 10 min -> ceil(2.2) = 3
    assert F.fleet(toy_city, toy_net, [10.0, 10.0]) == pytest.approx([5.0, 3.0])


# obe skale su donje granice, isto kao cost_scales u assignment.py
def test_scales_by_hand(toy_city):
    cp, flota = F.scales(toy_city)
    # putnicka: demand-ponderisano ulicno vreme (20 min) + pola najkraceg intervala
    assert cp == pytest.approx(20.0 + F.H_MIN / 2.0)
    # operaterska je veca od dve granice, pa zaokruzena navise:
    #   po kapacitetu 6000 * 0.10 / (80 * 60) = 0.125 vozila
    #   po mrezi      2 * MST(30) * 1.1 / 60  = 1.1 vozila   <- ova je veca
    assert flota == pytest.approx(2.0)


# kad su intervali poznati, ulazak kosta pola intervala umesto fiksnih 5 min
def test_assignment_with_headways_charges_half_headway(toy_city, toy_net):
    from tndp.core.assignment import assign

    res = assign(toy_city, toy_net, headways=[10.0, 20.0])
    # 0->2: cekanje na A (5) + voznja 0-1-2 (20) = 25
    assert res.travel_time[0, 2] == pytest.approx(25.0)
    # 0->3: cekanje na A (5) + voznja 0-1 (10) + cekanje na B (10) + voznja 1-3 (10) = 35
    assert res.travel_time[0, 3] == pytest.approx(35.0)


# opterecenje po liniji i po deonici; ovo do sada nije bilo pokriveno nijednim testom
def test_boardings_and_max_load_by_hand(toy_city, toy_net):
    from tndp.core.assignment import assign

    res = assign(toy_city, toy_net, compute_loads=True)
    # par 0-2 (100 u svakom smeru) ulazi samo u A: 200 ulazaka
    # par 0-3 (50 u svakom smeru) ulazi i u A i u B: A jos 100, B 100
    assert res.boardings == pytest.approx([300.0, 100.0])
    # najopterecenija deonica linije A je 0-1: nosi oba para, 200 + 100 = 300
    # deonica 1-2 nosi samo par 0-2, dakle 200, pa maksimum ostaje 300
    assert res.max_load == pytest.approx([300.0, 100.0])
    # nijedna deonica ne moze da nosi vise nego sto je u liniju uslo
    assert np.all(res.max_load <= res.boardings)


# petlja dodela -> opterecenje -> intervali -> dodela mora da stane
def test_evaluate_converges_and_matches_hand_numbers(toy_city, toy_net):
    # bez preskaliranja je toy grad premali, pa sve zavrsi na plafonu intervala
    o = F.evaluate(toy_city, toy_net, daily_trips=4800.0)
    # ratio 16: linija A ima 300 * 16 = 4800 opterecenja -> 10 min
    #           linija B ima 100 * 16 = 1600            -> 30 min
    assert o["h"] == pytest.approx([10.0, 30.0])
    # vozila: A ceil(44 / 10) = 5, B ceil(22 / 30) = 1
    assert o["vozila"] == pytest.approx([5.0, 1.0])
    assert o["flota"] == pytest.approx(6.0)
    # cekanje je pola intervala, ponderisano ulascima: (10*300 + 30*100)/400 / 2
    assert o["cekanje"] == pytest.approx(7.5)
    # jos jedan prolaz vise ne sme nista da promeni
    assert F.evaluate(toy_city, toy_net, daily_trips=4800.0, passes=8)["h"] \
        == pytest.approx(o["h"])


# linija bez putnika dobija najredji interval; to je uzrok dvopolne greske na
# Novom Sadu (results/novisad-frekvencije.md), pa mora da bude fiksirano testom
def test_empty_route_goes_to_max_headway():
    # cvor 4 je slepi krak bez ikakve traznje, pa linija 1-4 nema sta da preveze
    n = 5
    street = np.full((n, n), np.inf)
    np.fill_diagonal(street, 0.0)
    for a, b in [(0, 1), (1, 2), (1, 3), (1, 4)]:
        street[a, b] = street[b, a] = 10.0
    demand = np.zeros((n, n))
    demand[0, 2] = demand[2, 0] = 100.0
    demand[0, 3] = demand[3, 0] = 50.0
    city = CityGraph(coords=np.zeros((n, 2)), street_time=street, demand=demand, name="slepi")
    net = TransitNetwork(routes=[[0, 1, 2], [1, 3], [1, 4]])

    o = F.evaluate(city, net)
    assert o["res"].boardings[2] == pytest.approx(0.0)
    assert o["h"][2] == pytest.approx(F.H_MAX)


def test_objective_weights_terms_like_assignment_objective():
    scales_ = (10.0, 4.0)  # putnicki clan da 2.0, operaterski 1.0
    assert F.objective(20.0, 4.0, scales_, alpha=1.0) == pytest.approx(2.0)
    assert F.objective(20.0, 4.0, scales_, alpha=0.0) == pytest.approx(1.0)
    assert F.objective(20.0, 4.0, scales_, alpha=0.5) == pytest.approx(1.5)
