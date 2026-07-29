import numpy as np
import pytest

from tndp.synth.generator import generate_city


def test_generated_city_is_valid():
    for seed in range(5):
        city = generate_city(seed=seed)
        assert city.validate() == []
        assert 20 <= city.n <= 60
        assert np.allclose(city.demand, city.demand.T)
    # determinizam
    a = generate_city(seed=3)
    b = generate_city(seed=3)
    assert np.array_equal(a.demand, b.demand)
    assert np.array_equal(a.street_time, b.street_time)


def test_gravity_ima_prostornu_strukturu():
    # gravity mora biti koncentrisaniji od uniformnog: veći deo ukupnog
    # demanda u top 10% parova
    def top_share(city):
        vals = np.sort(city.demand[np.triu_indices(city.n, 1)])[::-1]
        k = max(1, len(vals) // 10)
        return vals[:k].sum() / vals.sum()

    g = top_share(generate_city(n=40, seed=1, demand_mode="gravity"))
    u = top_share(generate_city(n=40, seed=1, demand_mode="uniform"))
    assert g > u + 0.1
    # ukupan broj putovanja isti u oba režima
    assert generate_city(n=40, seed=1).demand.sum() == pytest.approx(
        generate_city(n=40, seed=1, demand_mode="uniform").demand.sum())


