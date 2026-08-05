# Mreže linija nacrtane na stvarnoj geografiji Novog Sada: granice mesnih
# zajednica iz mz.geojson kao podloga, zone kao tačke skalirane tražnjom,
# linije kao izlomljene putanje između težišta zona.
#
# viz/maps.py crta sintetičke gradove u apstraktnim koordinatama; ovde je
# podloga stvarna, pa se vidi da li mreža prati oblik grada — Dunav, Petrovaradin
# na desnoj obali, izduženost duž reke.
#
# pokretanje: python -m tndp.novisad.karta

import json
from pathlib import Path

import numpy as np

from tndp.baselines.hill_climb import hill_climb
from tndp.core.assignment import assign, cost_scales, objective
from tndp.experiments.common import load_policy
from tndp.novisad import konstante
from tndp.novisad.instanca import gsp_mreza, ucitaj
from tndp.novisad.ulice import ucitaj_zone
from tndp.rl.evaluate import decode_sampling
from tndp.viz import style

REZULTATI = Path(__file__).resolve().parent.parent.parent / "results"


def _granice(imena):
    fc = json.loads((konstante.DATA / "mz.geojson").read_text(encoding="utf-8"))
    trazene = set(imena)
    poligoni = []
    for f in fc["features"]:
        if f["properties"]["naziv"] not in trazene:
            continue
        g = f["geometry"]
        delovi = g["coordinates"] if g["type"] == "Polygon" else \
            [d for mp in g["coordinates"] for d in mp]
        for prsten in delovi:
            poligoni.append(np.array(prsten, dtype=float))
    return poligoni


# Težišta zona se čitaju iz zone.csv, ne rekonstruišu iz city.coords. Coords su
# centrirane i u kilometrima, pa bi vraćanje u stepene tražilo tačno težište
# područja — pogađanje te konstante pomera tačke u odnosu na granice iz geojsona.
def _tezista(zone):
    return (np.array([float(z["lon"]) for z in zone]),
            np.array([float(z["lat"]) for z in zone]))


def _nacrtaj(ax, city, lon, lat, granice, net, naslov, boja):
    for p in granice:
        ax.plot(p[:, 0], p[:, 1], lw=0.4, color="#cccccc", zorder=1)

    tez = city.demand.sum(1)
    vel = 12 + 90 * (tez - tez.min()) / (tez.max() - tez.min() + 1e-9)
    ax.scatter(lon, lat, s=vel, c="#444444", zorder=3, linewidths=0)

    # svaka linija dobija mali poprečni pomeraj da se preklopljene trase vide
    for k, r in enumerate(net.routes):
        d = (k - len(net.routes) / 2) * 0.00035
        ax.plot(lon[list(r)] + d, lat[list(r)] + d, lw=1.6, alpha=0.75,
                color=boja, zorder=2, solid_capstyle="round")

    ax.set_title(naslov, fontsize=10)
    ax.set_aspect(1 / np.cos(np.radians(float(lat.mean()))))
    ax.axis("off")


def main():
    import matplotlib.pyplot as plt

    city, imena = ucitaj()
    lon, lat = _tezista(ucitaj_zone())
    granice = _granice(imena)
    gsp, _ = gsp_mreza(city, imena)
    R = len(gsp.routes)
    duz = [len(r) for r in gsp.routes]
    lo, hi = min(duz), max(duz)
    scales = cost_scales(city)

    pol, cfg = load_policy("runs/novisad-r19/best.pt")
    mreze = [
        ("GSP, postojeća mreža", gsp, "#000000"),
        ("RL sampling 32", decode_sampling(pol, city, R, k=32, min_len=lo,
                                           max_len=hi, alpha=0.5)[0],
         style.color_for("RL sampling")),
        ("hill climbing", hill_climb(city, R, lo, hi, alpha=0.5)[0],
         style.color_for("hill climbing")),
    ]

    fig, axes = plt.subplots(1, len(mreze), figsize=(4.6 * len(mreze), 5.2))
    for ax, (ime, net, boja) in zip(axes, mreze):
        res = assign(city, net)
        pod = (f"{ime}\ncilj {objective(res, scales, 0.5):.2f}   "
               f"C_p {res.C_p:.1f} min   d_un {res.d['d_un']:.2f}")
        _nacrtaj(ax, city, lon, lat, granice, net, pod, boja)
    fig.suptitle(f"Novi Sad, {city.n} mesnih zajednica, R={R} linija", fontsize=12)
    fig.tight_layout()
    for p in style.save(fig, REZULTATI / "novisad-mreze"):
        print("->", p)


if __name__ == "__main__":
    main()
