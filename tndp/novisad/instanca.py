# graf Novog Sada u obliku koji model razume: 32 mesne zajednice su cvorovi

import csv
import re

import numpy as np
from scipy.sparse.csgraph import dijkstra

from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork
from tndp.novisad import konstante
from tndp.novisad.ulice import ucitaj_zone

# stepeni -> kilometri na geografskoj sirini Novog Sada, isto kao u traznja.py
KM_PO_STEPENU_LAT = 111.32
KM_PO_STEPENU_LON = 78.0


def _matrica(ime, imena):
    with open(konstante.DATA / ime, encoding="utf-8") as f:
        redovi = list(csv.reader(f))
    zaglavlje = redovi[0][1:]
    poredak = [zaglavlje.index(m) for m in imena]
    m = np.array([[float(x) for x in red[1:]] for red in redovi[1:]])
    return m[np.ix_(poredak, poredak)]


def _susedstvo(imena):
    mesto = {m: i for i, m in enumerate(imena)}
    adj = np.zeros((len(imena), len(imena)), dtype=bool)
    with open(konstante.DATA / "susedstvo.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["a"] in mesto and r["b"] in mesto:
                adj[mesto[r["a"]], mesto[r["b"]]] = True
                adj[mesto[r["b"]], mesto[r["a"]]] = True
    return adj


# CityGraph ocekuje koordinate u ravni; lokalna ekvidistantna projekcija je na ovoj velicini dovoljna (grad je ~15 km
def _koordinate(zone):
    lat = np.array([float(r["lat"]) for r in zone])
    lon = np.array([float(r["lon"]) for r in zone])
    xy = np.column_stack([lon * KM_PO_STEPENU_LON, lat * KM_PO_STEPENU_LAT])
    return xy - xy.mean(axis=0)


# tau.csv je najkrace vreme kroz celu ulicnu mrezu za SVAKI par zona, dakle metricki zatvarac
def ucitaj():
    zone = ucitaj_zone()
    imena = [r["mz"] for r in zone]
    n = len(imena)
    tau = _matrica("tau.csv", imena)
    adj = _susedstvo(imena)

    street = np.full((n, n), np.inf)
    street[adj] = tau[adj]
    np.fill_diagonal(street, 0.0)

    city = CityGraph(coords=_koordinate(zone), street_time=street,
                     demand=_matrica("traznja.csv", imena), name="NoviSad")
    return city, imena


# oznaka varijante -> osnovna linija: 1GL i 1J su varijante linije 1, 10APT linije 10, 18A i 18B linije 18
def _osnovna(oznaka):
    m = re.match(r"^\d+", oznaka)
    return m.group(0) if m else oznaka


def _stajaliste_u_zonu():
    with open(konstante.DATA / "stajalista_zone.csv", encoding="utf-8") as f:
        return {r["stajaliste_id"]: r["mz"] for r in csv.DictReader(f)
                if r["u_studiji"] == "1"}


# model gradi PROSTE puteve, a `ruta` u linije.csv je ceo kruzni tok linije: ista zona se javlja i u odlasku i u povratku
def _prost_put(niz):
    najbolji = (0, 0)
    poslednje = {}
    pocetak = 0
    for k, z in enumerate(niz):
        if z in poslednje and poslednje[z] >= pocetak:
            pocetak = poslednje[z] + 1
        poslednje[z] = k
        if k + 1 - pocetak > najbolji[1] - najbolji[0]:
            najbolji = (pocetak, k + 1)
    return niz[najbolji[0]:najbolji[1]]


# posle secenja petlji uzastopne zone ne moraju vise biti susedne; isto vazi i za zone koje trasa samo proseca bez
def _spoji(niz, street):
    pred = dijkstra(np.where(np.isfinite(street), street, 0.0),
                    directed=False, return_predecessors=True)[1]
    out = [niz[0]]
    for a, b in zip(niz, niz[1:]):
        if np.isfinite(street[a, b]):
            out.append(b)
            continue
        put, c = [], b
        while c != a and c >= 0:
            put.append(c)
            c = pred[a, c]
        out.extend(reversed(put))
    return out


def _u_zone(ruta, u_zonu, mesto):
    niz = []
    for s in ruta.split(";"):
        z = u_zonu.get(s)
        if z is not None and z in mesto and (not niz or niz[-1] != mesto[z]):
            niz.append(mesto[z])
    return niz


# rekonstrukcija postojece mreze: po jedna trasa za svaku od 19 osnovnih gradskih linija
def gsp_mreza(city, imena):
    mesto = {m: i for i, m in enumerate(imena)}
    u_zonu = _stajaliste_u_zonu()
    with open(konstante.DATA / "linije.csv", encoding="utf-8") as f:
        linije = [r for r in csv.DictReader(f) if r["tip"] == "gradska"]

    najbolja = {}
    for r in linije:
        sirovo = _u_zone(r["ruta"], u_zonu, mesto)
        if len(sirovo) < 2:
            continue
        k = _osnovna(r["oznaka"])
        if k not in najbolja or len(sirovo) > len(najbolja[k][0]):
            najbolja[k] = (sirovo, r)

    rute, dnevnik = [], []
    for k in sorted(najbolja, key=int):
        sirovo, r = najbolja[k]
        put = sirovo
        for _ in range(10):
            novi = _spoji(_prost_put(put), city.street_time)
            if novi == put:
                break
            put = novi
        put = _prost_put(put)
        rute.append(put)
        dnevnik.append({"linija": k, "varijanta": r["oznaka"],
                        "zona_sirovo": len(sirovo), "zona_trasa": len(put),
                        "umetnuto": len([z for z in put if z not in sirovo]),
                        "izbaceno": len([z for z in sirovo if z not in put]),
                        "naziv": r["naziv"]})
    return TransitNetwork(routes=rute), dnevnik


def main():
    city, imena = ucitaj()
    print(f"graf: {city.n} zona, {len(city.street_edges)} uličnih ivica, "
          f"prosečan stepen {2 * len(city.street_edges) / city.n:.1f}")
    problemi = city.validate()
    print("validate():", problemi or "bez prekršaja")

    tau = _matrica("tau.csv", imena)
    gore = np.triu_indices(city.n, 1)
    odnos = city.street_shortest[gore] / np.maximum(tau[gore], 1e-9)
    print(f"put kroz zonski graf vs tau po uličnoj mreži: medijana "
          f"{np.median(odnos):.3f}, maksimum {odnos.max():.3f} "
          f"(1.0 = zonski graf ne produžava putovanje)")
    print(f"MST {city.mst_time:.1f} min, donja granica putničkog vremena "
          f"{city.street_shortest_mean_demand:.2f} min")

    mreza, dnevnik = gsp_mreza(city, imena)
    print(f"\nGSP mreža: {len(mreza.routes)} osnovnih gradskih linija")
    print(f"{'linija':>6} {'varijanta':>9} {'sirovo':>7} {'trasa':>6} "
          f"{'umetnuto':>9} {'izbačeno':>9}  naziv")
    for d in dnevnik:
        print(f"{d['linija']:>6} {d['varijanta']:>9} {d['zona_sirovo']:7d} "
              f"{d['zona_trasa']:6d} {d['umetnuto']:9d} {d['izbaceno']:9d}  "
              f"{d['naziv'][:40]}")

    duz = [len(r) for r in mreza.routes]
    print(f"\ndužina linije u zonama: min {min(duz)}, medijana "
          f"{int(np.median(duz))}, maksimum {max(duz)}")
    pokrivene = {v for r in mreza.routes for v in r}
    print(f"zona na bar jednoj liniji: {len(pokrivene)} od {city.n}")
    if len(pokrivene) < city.n:
        print("  bez linije:", [imena[i] for i in range(city.n)
                                if i not in pokrivene])
    print("prekršaji mreže:", mreza.check(city) or "nema")


if __name__ == "__main__":
    main()
