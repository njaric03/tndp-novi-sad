"""Acceptance test faze 1: nasa cost funkcija na objavljenim Mandl mrezama.

Referentne vrednosti su iz Holliday et al. (arXiv:2404.05894, tabele 3 i 4),
uz standardni transfer penal od 5 minuta:
  - best passenger (Mumford 2013): ATT 10.27 min, ukupno vreme linija 221 min
  - best operator: ukupno vreme linija 63 min; za ATT tabela navodi 15.13
    (Mumford 2013) odnosno 13.48 (John et al. 2014) za mreze sa istim C_o.
    Nas standardni shortest-path assignment nad arhiviranim route setom daje
    tacno 13.48, sto se poklapa sa John et al. konvencijom; Mumford je svoj
    ATT racunala sopstvenom procedurom assignmenta.
"""

from pathlib import Path

import pytest

from tndp.core.assignment import assign
from tndp.core.io import load_benchmark_city, load_literature_solutions
from tndp.core.network import TransitNetwork

DATA = Path(__file__).parent.parent / "data" / "benchmarks" / "Mandl" / "Mandl1"


@pytest.fixture(scope="module")
def mandl():
    city = load_benchmark_city(DATA)
    assert city.validate() == []
    return city


@pytest.fixture(scope="module")
def solutions():
    return load_literature_solutions(
        DATA / "literature_solutions_for_mandl1_20181025.txt"
    )


def test_mandl_basic_properties(mandl):
    assert mandl.n == 15
    assert len(mandl.street_edges) == 21
    assert mandl.demand.sum() == 15570


def test_mumford2013_best_passenger(mandl, solutions):
    net = TransitNetwork(routes=solutions["Mumford (2013) 6 best passenger"])
    assert net.check(mandl, num_routes=6, min_len=2, max_len=8) == []
    res = assign(mandl, net)
    assert res.is_connected
    assert res.C_p == pytest.approx(10.27, abs=0.02)
    assert res.C_o == pytest.approx(221, abs=0.5)


def test_mumford2013_best_operator(mandl, solutions):
    net = TransitNetwork(routes=solutions["Mumford (2013) 6 best operator"])
    res = assign(mandl, net)
    assert res.is_connected
    assert res.C_o == pytest.approx(63, abs=0.5)
    assert res.C_p == pytest.approx(13.48, abs=0.02)
