import numpy as np

from tndp.core.assignment import assign

# Dodela frekvencija je druga faza problema: trase su date, trazi se koliko cesto svaka linija vozi

# putnika po vozilu koje se racuna kao pun kapacitet
CAPACITY = 80.0
# udeo dnevne traznje koji padne u vrsni sat
PEAK_SHARE = 0.10
# politika prevoznika: ispod 5 min se ne planira, iznad 60 min linija prakticno ne postoji kao gradska
H_MIN, H_MAX = 5.0, 60.0
# obrtno vreme na okretnici kao udeo vremena voznje
LAYOVER = 0.10


# interval sledjenja iz najopterecenije deonice: toliko vozila na sat treba da vrh stane u kapacitet
def headways(max_load, capacity=CAPACITY, peak_share=PEAK_SHARE):
    peak = np.asarray(max_load, dtype=float) * peak_share
    per_hour = peak / capacity
    with np.errstate(divide="ignore"):
        h = np.where(per_hour > 0, 60.0 / np.maximum(per_hour, 1e-9), H_MAX)
    return np.clip(h, H_MIN, H_MAX)


# broj vozila po liniji: obilazak u oba smera plus obrt, podeljeno intervalom
def fleet(city, network, h):
    cycle = 2.0 * network.route_times(city) * (1.0 + LAYOVER)
    return np.ceil(cycle / np.asarray(h, dtype=float))


# donje granice istog tipa kao cp_scale i mst_time u assignment.py, da bi alpha i ovde balansirala
def scales(city, daily_trips=None):
    ratio = 1.0 if daily_trips is None else daily_trips / city.demand.sum()
    cp = city.street_shortest_mean_demand + H_MIN / 2.0
    passenger_minutes = (city.demand * city.street_shortest).sum() * ratio * PEAK_SHARE
    by_capacity = passenger_minutes / (CAPACITY * 60.0)
    by_network = 2.0 * city.mst_time * (1.0 + LAYOVER) / H_MAX
    return cp, float(max(np.ceil(max(by_capacity, by_network)), 1.0))


def objective(C_p_all, vehicles, scales_, alpha=0.5):
    cp_scale, fleet_scale = scales_
    return alpha * C_p_all / cp_scale + (1 - alpha) * vehicles / fleet_scale


# dve faze u petlji: dodela -> opterecenje -> intervali -> dodela
def evaluate(city, network, alpha=0.5, passes=3, capacity=CAPACITY,
             peak_share=PEAK_SHARE, daily_trips=None):
    ratio = 1.0 if daily_trips is None else daily_trips / city.demand.sum()

    # prvi prolaz jos ne zna intervale, pa ulazak kosta fiksni penal iz literature
    res = assign(city, network, compute_loads=True)
    h = headways(res.max_load * ratio, capacity, peak_share)
    for _ in range(passes):
        res = assign(city, network, compute_loads=True, headways=h)
        h_new = headways(res.max_load * ratio, capacity, peak_share)
        if np.allclose(h_new, h):
            break  # res je vec racunat sa ovim h, nema sta da se ponavlja
        h = h_new
    else:
        # petlja je istrosila prolaze bez konvergencije: h se promenilo posle
        # poslednje dodele, pa res mora jos jednom da se uskladi sa njim
        res = assign(city, network, compute_loads=True, headways=h)
    vehicles = fleet(city, network, h)
    return {
        "res": res,
        "h": h,
        "vozila": vehicles,
        "flota": float(vehicles.sum()),
        "cilj": objective(res.C_p_all, float(vehicles.sum()),
                          scales(city, daily_trips), alpha),
        "cekanje": float(np.average(h, weights=np.maximum(res.boardings, 1e-9)) / 2.0),
    }
