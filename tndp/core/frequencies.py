import numpy as np

from tndp.core.assignment import assign

# Dodela frekvencija je druga faza problema: trase su date, traži se koliko
# često svaka linija vozi. Prva faza (izbor trasa) je ono što RL uči; ovde se
# nad gotovom mrežom računa ono što se bez frekvencija ne može izraziti —
# vreme čekanja putnika i broj vozila koji operateru stvarno treba.
#
# Zavisnost je kružna: interval sleđenja se određuje iz opterećenja, a
# opterećenje zavisi od dodele koja zavisi od intervala (putnik bira brzu
# liniju, a brza je ona koja češće vozi). Rešava se u par prolaza, `oceni`
# ispod, i konvergira brzo jer promena intervala retko menja izbor rute.

# putnika po vozilu koje se računa kao pun kapacitet. GSP Novi Sad vozi
# mešavinu solo i zglobnih vozila; 80 je red veličine solo gradskog autobusa
# sa mestima za stajanje. osetljivost je u bench_freq.
KAPACITET = 80.0
# udeo dnevne tražnje koji padne u vršni sat. planerska konvencija je 8-12%;
# frekvencija se dimenzioniše po vrhu, ne po proseku.
UDEO_VRHA = 0.10
# politika prevoznika: ispod 5 min se ne planira, iznad 60 min linija
# praktično ne postoji kao gradska
H_MIN, H_MAX = 5.0, 60.0
# obrtno vreme na okretnici kao udeo vremena vožnje
OBRT = 0.10


# interval sleđenja iz najopterećenije deonice: toliko vozila na sat treba da
# vrh stane u kapacitet. van tog ograničenja radi politika min/max.
def intervali(max_load, kapacitet=KAPACITET, udeo_vrha=UDEO_VRHA):
    vrh = np.asarray(max_load, dtype=float) * udeo_vrha
    vozila_na_sat = vrh / kapacitet
    with np.errstate(divide="ignore"):
        h = np.where(vozila_na_sat > 0, 60.0 / np.maximum(vozila_na_sat, 1e-9), H_MAX)
    return np.clip(h, H_MIN, H_MAX)


# broj vozila po liniji: obilazak u oba smera plus obrt, podeljeno intervalom.
# ovo je trošak operatera koji `C_o` (vreme vožnje u jednom smeru) ne vidi —
# duga retka linija i kratka česta mogu imati isti C_o a različitu flotu.
def flota(city, network, h):
    obilazak = 2.0 * network.route_times(city) * (1.0 + OBRT)
    return np.ceil(obilazak / np.asarray(h, dtype=float))


# donje granice istog tipa kao cp_scale i mst_time u assignment.py, da bi
# alpha i ovde balansirala. putnik u najboljem slučaju vozi ulično najkraće
# vreme uz najmanje dozvoljeno čekanje. za flotu se uzima veća od dve donje
# granice:
#   - kapacitetska: u vršnom satu treba prevesti toliko putnik-minuta, a
#     jedno vozilo isporučuje KAPACITET putnik-minuta po minutu vožnje;
#     dostiže se samo ako je svako vozilo puno celo vreme i vozi najkraćim
#     uličnim putem,
#   - mrežna: mreža oblika MST-a vožena najređim dozvoljenim intervalom.
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


# dve faze u petlji: dodela -> opterećenje -> intervali -> dodela. prvi prolaz
# ide sa fiksnim penalom iz literature jer intervali još ne postoje.
#
# `putovanja_dnevno` postoji zato što matrica tražnje nosi broj putovanja ali
# ne i period na koji se odnosi. Za Novi Sad je period poznat (brojanje 2017,
# radni dan), pa se ne prosleđuje ništa. Mandl i Mumford ne objavljuju period,
# a bez njega je vršni sat neodređen i intervali se zalepe za H_MIN ili H_MAX;
# tamo se matrica preskalira na navedeni dnevni obim i to MORA da piše uz
# rezultat, jer je pretpostavka a ne podatak.
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
