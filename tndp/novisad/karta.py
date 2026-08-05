# Mreze linija nacrtane preko stvarne ulicne mreze Novog Sada
# podloga je OSM graf ulica i granice mesnih zajednica, vidi novisad/podloga.py
# pokretanje: python -m tndp.novisad.karta

from pathlib import Path

import numpy as np

from tndp.baselines.hill_climb import hill_climb
from tndp.core.assignment import assign, cost_scales, objective
from tndp.experiments.common import load_policy
from tndp.novisad import podloga
from tndp.novisad.instanca import gsp_mreza, ucitaj
from tndp.novisad.ulice import ucitaj_zone
from tndp.rl.evaluate import decode_sampling
from tndp.viz import style

KOREN = Path(__file__).resolve().parent.parent.parent
REZULTATI = KOREN / "results"
MODEL = KOREN / "runs" / "novisad-r19" / "best.pt"


# tezista zona se citaju iz zone.csv, ne rekonstruisu iz centriranih koordinata
def _tezista(zone):
    return (np.array([float(z["lon"]) for z in zone]),
            np.array([float(z["lat"]) for z in zone]))


def _panel(ax, city, imena, lon, lat, put, net, naslov, podnaslov, boja, prvi=False):
    # zone su obojene traznjom; traznja je svojstvo zone a ne tacke u njoj
    podloga.nacrtaj(ax, imena, vrednosti=city.demand.sum(1))

    # trasa prati ulice: prava linija izmedju tezista implicira ulicu koje nema
    for k, r in enumerate(net.routes):
        if len(r) < 2:
            continue
        pom = (k - len(net.routes) / 2) * 0.00022
        delovi = [put(a, b) for a, b in zip(r, r[1:])]
        xy = np.vstack(delovi)
        ax.plot(xy[:, 0] + pom, xy[:, 1] + pom, lw=3.2, color="white", alpha=0.8,
                zorder=3, solid_capstyle="round", solid_joinstyle="round")
        ax.plot(xy[:, 0] + pom, xy[:, 1] + pom, lw=1.6, color=boja, alpha=0.92,
                zorder=4, solid_capstyle="round", solid_joinstyle="round")

    # cvorovi trase, tamo gde linija stvarno staje
    u_mrezi = sorted({v for r in net.routes for v in r})
    ax.scatter(lon[u_mrezi], lat[u_mrezi], s=13, c="#1a1a1a", zorder=5,
               linewidths=0.7, edgecolors="white")

    # kadar se racuna od tezista zona; granice se pruzaju juzno gde linija nema
    mx = (lon.max() - lon.min()) * 0.16
    my = (lat.max() - lat.min()) * 0.16
    x0, x1 = lon.min() - mx, lon.max() + mx
    y0, y1 = lat.min() - my, lat.max() + my
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.radians(float(lat.mean()))))
    ax.axis("off")
    ax.set_title(naslov, fontsize=11, pad=8, weight="bold")
    ax.text(0.5, -0.035, podnaslov, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.5, color="#444444")
    if prvi:
        podloga.razmernik(ax, x0, x1, y0 + (y1 - y0) * 0.03)


def main(checkpoint=MODEL):
    import matplotlib.pyplot as plt

    city, imena = ucitaj()
    lon, lat = _tezista(ucitaj_zone())
    put = podloga.trase(lon, lat)
    gsp, _ = gsp_mreza(city, imena)
    R = len(gsp.routes)
    duz = [len(r) for r in gsp.routes]
    lo, hi = min(duz), max(duz)
    scales = cost_scales(city)

    pol, _ = load_policy(checkpoint)
    mreze = [
        ("GSP, postojeća mreža", gsp, "#1a1a1a"),
        ("RL politika, sampling 32",
         decode_sampling(pol, city, R, k=32, min_len=lo, max_len=hi, alpha=0.5)[0],
         style.color_for("RL sampling")),
        ("hill climbing", hill_climb(city, R, lo, hi, alpha=0.5)[0],
         style.color_for("hill climbing")),
    ]

    style.primeni()
    fig, axes = plt.subplots(1, len(mreze), figsize=(4.9 * len(mreze), 5.6))
    for i, (ax, (ime, net, boja)) in enumerate(zip(axes, mreze)):
        res = assign(city, net)
        pod = (f"cilj {objective(res, scales, 0.5):.2f}    "
               f"C_p {res.C_p:.1f} min    "
               f"bez veze {100 * res.d['d_un']:.0f}%")
        _panel(ax, city, imena, lon, lat, put, net, ime, pod, boja, prvi=(i == 0))

    fig.suptitle(f"Novi Sad: {city.n} mesnih zajednica, {R} linija",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.005, "trase prate stvarnu uličnu mrežu; zasićenost zone je njena "
             "ukupna tražnja; tačka je zona kroz koju linija prolazi",
             ha="center", fontsize=8, color="#777777")
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])
    for p in style.save(fig, REZULTATI / "novisad-mreze"):
        print("->", p)


if __name__ == "__main__":
    main()
