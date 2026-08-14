# Instanca Novog Sada: da li sredjeni podaci jos daju graf i mrezu kakve
# rezultati u results/ pretpostavljaju.
#
# data/novisad/ je u .gitignore, pravi ga tndp.novisad.preuzmi + sredi. Bez tih
# fajlova ceo modul se preskace, da suite prolazi i na svezem kloniranju.

import pytest

from tndp.novisad import konstante

POTREBNI = ["zone.csv", "tau.csv", "susedstvo.csv", "traznja.csv",
            "linije.csv", "stajalista_zone.csv"]

pytestmark = pytest.mark.skipif(
    not all((konstante.DATA / f).exists() for f in POTREBNI),
    reason="data/novisad/ nije sagrađen (python -m tndp.novisad.preuzmi && ... sredi)")


@pytest.fixture(scope="module")
def instanca():
    from tndp.novisad.instanca import ucitaj
    return ucitaj()


@pytest.fixture(scope="module")
def gsp(instanca):
    from tndp.novisad.instanca import gsp_mreza
    city, imena = instanca
    return gsp_mreza(city, imena)


# 32 zone je broj na kom stoje svi brojevi u results/novisad-*.md
def test_zonski_graf_je_validan(instanca):
    city, imena = instanca
    assert city.n == 32
    assert len(imena) == city.n
    assert city.validate() == []


def test_traznja_je_simetricna_i_bez_dijagonale(instanca):
    import numpy as np

    from tndp.novisad.traznja import PUTOVANJA
    city, _ = instanca
    assert np.allclose(city.demand, city.demand.T)
    assert np.allclose(np.diag(city.demand), 0.0)
    # zbir matrice su PUTOVANJA, ne 172.687 voznji iz brojanja (traznja.py: /1.779)
    assert city.demand.sum() == pytest.approx(PUTOVANJA, rel=0.01)


# R=19 i duzina [2, 14] su vrednosti iz configs/novisad-r19.yaml; ako se
# rekonstrukcija trase pomeri, config vise ne odgovara instanci
def test_gsp_mreza_ima_19_linija_u_granicama_configa(gsp, instanca):
    city, _ = instanca
    mreza, dnevnik = gsp
    assert len(mreza.routes) == 19
    assert len(dnevnik) == 19
    duzine = [len(r) for r in mreza.routes]
    assert min(duzine) == 2 and max(duzine) == 14


def test_gsp_mreza_prolazi_iste_provere_kao_resenja_metoda(gsp, instanca):
    city, _ = instanca
    mreza, _ = gsp
    # ista provera koju u poredjenju mora da prodje i svaka metoda
    assert mreza.check(city, num_routes=19, min_len=2, max_len=14) == []


# poredjenje sa GSP-om nema smisla ako neka zona nije ni na jednoj liniji
def test_gsp_mreza_pokriva_sve_zone(gsp, instanca):
    city, _ = instanca
    mreza, _ = gsp
    pokrivene = {v for r in mreza.routes for v in r}
    assert len(pokrivene) == city.n


# trase su proste putanje: model gradi takve, pa i referenca mora biti takva
def test_trase_su_proste_putanje_po_susednim_zonama(gsp, instanca):
    import numpy as np
    city, _ = instanca
    mreza, _ = gsp
    for i, ruta in enumerate(mreza.routes):
        assert len(set(ruta)) == len(ruta), f"linija {i} ponavlja zonu"
        for a, b in zip(ruta, ruta[1:]):
            assert np.isfinite(city.street_time[a, b]), f"linija {i}: {a}-{b} nije ivica"


# frekvencijska faza je jedina spoljna provera u repou; mora bar da se izvrti
def test_frekvencijska_faza_daje_intervale_u_granicama(gsp, instanca):
    import numpy as np
    from tndp.core import frequencies as F
    city, _ = instanca
    mreza, _ = gsp
    o = F.evaluate(city, mreza)
    assert len(o["h"]) == 19
    assert np.all(o["h"] >= F.H_MIN) and np.all(o["h"] <= F.H_MAX)
    assert o["flota"] > 0 and np.isfinite(o["cekanje"])
