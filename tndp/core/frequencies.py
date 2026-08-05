import numpy as np

from tndp.core.assignment import assign

# Dodela frekvencija je druga faza problema: trase su date, traži se koliko često svaka linija vozi

# putnika po vozilu koje se računa kao pun kapacitet
KAPACITET = 80.0
# udeo dnevne tražnje koji padne u vršni sat
UDEO_VRHA = 0.10
# politika prevoznika: ispod 5 min se ne planira, iznad 60 min linija praktično ne postoji kao gradska
H_MIN, H_MAX = 5.0, 60.0
# obrtno vreme na okretnici kao udeo vremena vožnje
OBRT = 0.10


# interval sleđenja iz najopterećenije deonice: toliko vozila na sat treba da vrh stane u kapacitet
def intervali(max_load, kapacitet=KAPACITET, udeo_vrha=UDEO_VRHA):
    vrh = np.asarray(max_load, dtype=float) * udeo_vrha
    vozila_na_sat = vrh / kapacitet
    with np.errstate(divide="ignore"):
        h = np.where(vozila_na_sat > 0, 60.0 / np.maximum(vozila_na_sat, 1e-9), H_MAX)
    return np.clip(h, H_MIN, H_MAX)


# broj vozila po liniji: obilazak u oba smera plus obrt, podeljeno intervalom
def flota(city, network, h):
    obilazak = 2.0 * network.route_times(city) * (1.0 + OBRT)
    return np.ceil(obilazak / np.asarray(h, dtype=float))


# donje granice istog tipa kao cp_scale i mst_time u assignment.py, da bi alpha i ovde balansirala
def skale(city, putovanja_dnevno=None):
    razmera = 1.0 if putovanja_dnevno is None else putovanja_dnevno / city.demand.sum()
    cp = city.street_shortest_mean_demand + H_MIN / 2.0
    putnik_min = (city.demand * city.street_shortest).sum() * razmera * UDEO_VRHA
    kapacitetska = putnik_min / (KAPACITET * 60.0)
    mrezna = 2.0 * city.mst_time * (1.0 + OBRT) / H_MAX
    return cp, float(max(np.ceil(max(kapacitetska, mrezna)), 1.0))


def cilj(C_p_all, vozila, skale_, alpha=0.5):
    cp_scale, flota_scale = skale_
    return alpha * C_p_all / cp_scale + (1 - alpha) * vozila / flota_scale


# dve faze u petlji: dodela -> opterećenje -> intervali -> dodela
def oceni(city, network, alpha=0.5, prolaza=3, kapacitet=KAPACITET,
          udeo_vrha=UDEO_VRHA, putovanja_dnevno=None):
    razmera = 1.0 if putovanja_dnevno is None else putovanja_dnevno / city.demand.sum()

    res = assign(city, network, compute_loads=True)
    h = intervali(res.max_load * razmera, kapacitet, udeo_vrha)
    for _ in range(prolaza):
        res = assign(city, network, compute_loads=True, headways=h)
        h_novo = intervali(res.max_load * razmera, kapacitet, udeo_vrha)
        if np.allclose(h_novo, h):
            h = h_novo
            break
        h = h_novo
    res = assign(city, network, compute_loads=True, headways=h)
    vozila = flota(city, network, h)
    return {
        "res": res,
        "h": h,
        "vozila": vozila,
        "flota": float(vozila.sum()),
        "cilj": cilj(res.C_p_all, float(vozila.sum()),
                     skale(city, putovanja_dnevno), alpha),
        "cekanje": float(np.average(h, weights=np.maximum(res.boardings, 1e-9)) / 2.0),
    }
