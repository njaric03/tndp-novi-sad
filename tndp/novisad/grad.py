import csv
import re

import numpy as np
from scipy.sparse.csgraph import dijkstra

from tndp.core.city import CityGraph
from tndp.core.network import TransitNetwork
from tndp.novisad import izvori, traznja
from tndp.novisad.ulice import ucitaj_zone


# indeks zone po imenu mesne zajednice, redosledom iz zone.csv — isti redosled
# koriste tau.csv i traznja.csv
def _indeksi():
    zone = ucitaj_zone()
    return zone, {r["mz"]: i for i, r in enumerate(zone)}


def _matrica(ime, zone):
    with open(izvori.DATA / ime, encoding="utf-8") as f:
        redovi = list(csv.reader(f))
    poredak = [redovi[0][1:].index(r["mz"]) for r in zone]
    m = np.array([[float(x) for x in red[1:]] for red in redovi[1:]])
    return m[np.ix_(poredak, poredak)]


# CityGraph Novog Sada. Ulične ivice su parovi zona koje se stvarno graniče
# (susedstvo.csv), sa vremenom vožnje iz tau.csv; sve ostalo je inf, kao i kod
# sintetike. Time gustina grafa ostaje u opsegu na kom je politika trenirana —
# potpuna matrica bi svaki par proglasila direktnom vezom i problem bi nestao.
def izgradi(beta=traznja.BETA, ukupno=None):
    zone, idx = _indeksi()
    n = len(zone)
    tau = _matrica("tau.csv", zone)

    street = np.full((n, n), np.inf)
    with open(izvori.DATA / "susedstvo.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i, j = idx[r["a"]], idx[r["b"]]
            street[i, j] = street[j, i] = tau[i, j]
    np.fill_diagonal(street, 0.0)

    _, d = traznja.izgradi(beta=beta, ukupno=ukupno)

    # koordinate u kilometrima, ista konverzija kao u traznja._rastojanja
    lat = np.array([float(r["lat"]) for r in zone])
    lon = np.array([float(r["lon"]) for r in zone])
    coords = np.stack([(lon - lon.mean()) * 78.0, (lat - lat.mean()) * 111.32], axis=1)

    grad = CityGraph(coords=coords, street_time=street, demand=d, name="Novi Sad")
    problemi = grad.validate()
    assert problemi == [], problemi
    return grad, [r["mz"] for r in zone]


# osnovna oznaka linije: "10APT" i "10MAL" su varijante linije 10, a brojanje
# iz 2017 meri liniju kao celinu. "11A" i "11B" su dva kraka linije 11 i
# brojanje ih takođe daje zajedno.
def osnovna(oznaka):
    m = re.match(r"\d+", oznaka)
    return m.group(0) if m else oznaka


# GSP mreža prevedena u nizove zona. Uzastopna stajališta u istoj zoni se
# sažimaju, stajališta van područja studije ispadaju. Ostane li posle toga
# manje od `min_zona` zona, linija nije ruta u zonskom grafu nego tačka i
# izbacuje se — to je ograničenje zonskog pristupa, ne greška u podacima.
def gsp_mreza(grad, imena, min_zona=2):
    idx = {z: i for i, z in enumerate(imena)}
    sz = {}
    with open(izvori.DATA / "stajalista_zone.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sz[r["stajaliste_id"]] = r["mz"]

    # najkraći putevi po zonskom grafu, za popunjavanje skokova
    sp = dijkstra(np.where(np.isfinite(grad.street_time), grad.street_time, 0.0),
                  directed=False, return_predecessors=True)[1]

    rute, oznake, odbacene = [], [], []
    with open(izvori.DATA / "linije.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tip"] != "gradska" or r["varijanta"] != "0":
                continue
            seq = []
            for s in r["ruta"].split(";"):
                z = sz.get(s)
                if z is None or z not in idx:
                    continue
                if not seq or seq[-1] != idx[z]:
                    seq.append(idx[z])
            if len(seq) < min_zona:
                odbacene.append((r["oznaka"], r["smer"], len(seq)))
                continue
            rute.append(_popuni(seq, sp))
            oznake.append(r["oznaka"])
    return TransitNetwork(rute), oznake, odbacene


# dve uzastopne zone linije ne moraju biti susedne (stajališta između njih su
# u zoni van studije, ili linija preseca zonu bez stajališta). Umeće se
# najkraći put po zonskom grafu da bi niz bio putanja u uličnom grafu.
def _popuni(seq, pred):
    puna = [seq[0]]
    for a, b in zip(seq, seq[1:]):
        if not np.isfinite(pred[a, b]) and pred[a, b] < 0:
            puna.append(b)
            continue
        deo, cur = [], b
        while cur != a and cur >= 0:
            deo.append(cur)
            cur = pred[a, cur]
        puna += deo[::-1] if cur == a else [b]
    return puna
