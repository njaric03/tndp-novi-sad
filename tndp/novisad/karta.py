# Mreze linija nacrtane preko stvarne ulicne mreze Novog Sada
# podloga je OSM graf ulica i granice mesnih zajednica, vidi novisad/podloga.py
# pokretanje: python -m tndp.novisad.karta

from pathlib import Path

import matplotlib.patheffects as pe
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


FONT_NATPIS = 8.0

# Kandidati za mesto natpisa u odnosu na tacku zone, redom po pozeljnosti.
# Iznad tacke je najcitljivije, pa strane, pa udaljeniji redovi.
MESTA = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2),
         (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 3), (0, -3), (2, 0), (-2, 0)]


# Imena 32 zone na jednoj karti se sudaraju oko centra, gde su zone najmanje.
# Mesto se bira probanjem: natpis se nacrta, izmeri mu se STVARNI okvir preko
# renderera, i ako se seče sa vec postavljenim, briše se i proba sledeće mesto.
# Procena širine iz broja slova ne radi, jer slova nisu iste širine.
def _natpisi(ax, imena, lon, lat):
    fig = ax.figure
    fig.canvas.draw()
    render = fig.canvas.get_renderer()
    korak = FONT_NATPIS * 1.35

    def nacrtaj(i, dx, dy):
        return ax.annotate(
            imena[i], (lon[i], lat[i]), fontsize=FONT_NATPIS, zorder=6,
            ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
            va="center" if dy == 0 else ("bottom" if dy > 0 else "top"),
            color="#1a1a1a", xytext=(korak * dx * 0.4, korak * dy * 0.75),
            textcoords="offset points",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # koliko se okvir preklapa sa vec postavljenim natpisima, u kvadratnim
    # tackama; nula znaci slobodno mesto
    def preklop(bb, zauzeto):
        uk = 0.0
        for z in zauzeto:
            w = min(bb.x1, z.x1) - max(bb.x0, z.x0)
            h = min(bb.y1, z.y1) - max(bb.y0, z.y0)
            if w > 0 and h > 0:
                uk += w * h
        return uk

    zauzeto = []
    # najpre zone sa najduzim imenom: njima je najteze naci mesto
    for i in sorted(range(len(imena)), key=lambda k: -len(imena[k])):
        najbolje, najmanji = None, None
        for dx, dy in MESTA:
            t = nacrtaj(i, dx, dy)
            # razmak u tackama, ne u procentima: natpis nosi belu konturu koja
            # izlazi van okvira teksta, pa dva natpisa koja se tek dodiruju
            # svejedno pojedu jedan drugom kraj
            bb = t.get_window_extent(render).padded(3.0)
            p = preklop(bb, zauzeto)
            if najmanji is None or p < najmanji:
                najbolje, najmanji = (dx, dy, bb), p
            t.remove()
            if p == 0.0:
                break
        # ako nijedno mesto nije potpuno slobodno, uzima se ono sa najmanjim
        # preklapanjem, umesto da natpis padne na podrazumevano mesto
        dx, dy, bb = najbolje
        nacrtaj(i, dx, dy)
        zauzeto.append(bb)


def _panel(ax, city, imena, lon, lat, put, gustina, net, naslov, podnaslov, boja,
           prvi=False, lw=1.6, pad=0.16):
    # zone su obojene GUSTINOM traznje, ne ukupnom: ukupna bi tamnim obojila
    # velike periferne zone prosto zato sto su velike, a to je artefakt povrsine
    podloga.nacrtaj(ax, imena, vrednosti=gustina)

    # trasa prati ulice: prava linija izmedju tezista implicira ulicu koje nema
    for k, r in enumerate(net.routes):
        if len(r) < 2:
            continue
        pom = (k - len(net.routes) / 2) * 0.00022
        delovi = [put(a, b) for a, b in zip(r, r[1:])]
        xy = np.vstack(delovi)
        ax.plot(xy[:, 0] + pom, xy[:, 1] + pom, lw=2 * lw, color="white", alpha=0.8,
                zorder=3, solid_capstyle="round", solid_joinstyle="round")
        ax.plot(xy[:, 0] + pom, xy[:, 1] + pom, lw=lw, color=boja, alpha=0.92,
                zorder=4, solid_capstyle="round", solid_joinstyle="round")

    # cvorovi trase, tamo gde linija stvarno staje
    u_mrezi = sorted({v for r in net.routes for v in r})
    ax.scatter(lon[u_mrezi], lat[u_mrezi], s=13 if lw < 2 else 22,
               c="#1a1a1a", zorder=5, linewidths=0.7, edgecolors="white")

    # kadar se racuna od tezista zona; granice se pruzaju juzno gde linija nema
    mx = (lon.max() - lon.min()) * pad
    my = (lat.max() - lat.min()) * pad
    x0, x1 = lon.min() - mx, lon.max() + mx
    y0, y1 = lat.min() - my, lat.max() + my
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.radians(float(lat.mean()))))
    ax.axis("off")
    ax.set_title(naslov, fontsize=13, pad=8, weight="bold")
    ax.text(0.5, -0.035, podnaslov, transform=ax.transAxes, ha="center",
            va="top", fontsize=10.5, color="#444444")
    if prvi:
        podloga.razmernik(ax, x0, x1, y0 + (y1 - y0) * 0.03)


def main(checkpoint=MODEL):
    import matplotlib.pyplot as plt

    city, imena = ucitaj()
    zone = ucitaj_zone()
    lon, lat = _tezista(zone)
    put = podloga.trase(lon, lat)
    povrsina = np.array([float(z["povrsina_km2"]) for z in zone])
    gustina = city.demand.sum(1) / np.maximum(povrsina, 0.05)
    gsp, _ = gsp_mreza(city, imena)
    R = len(gsp.routes)
    duz = [len(r) for r in gsp.routes]
    lo, hi = min(duz), max(duz)
    scales = cost_scales(city)

    pol, _ = load_policy(checkpoint)
    rl = decode_sampling(pol, city, R, k=32, min_len=lo, max_len=hi, alpha=0.5)[0]
    hc = hill_climb(city, R, lo, hi, alpha=0.5)[0]

    def podnaslov(net):
        res = assign(city, net)
        return (f"cilj {objective(res, scales, 0.5):.2f}    "
                f"$C_p$ {res.C_p:.1f} min    "
                f"$C_o$ {res.C_o:.0f} min")

    # objasnjenje sta je sta ide u potpis figure u radu, ne na sliku: tekst
    # preko cele sirine bi bio najsiri element, pa bbox_inches="tight" vise
    # ne bi mogao da skrati praznine oko karte i karta bi u radu ispala mala
    style.primeni()

    # Glavni rezultat studije slucaja ide sam i veliki, sa imenima zona.
    # Na tri panela jedan uz drugi se mreza politike ne moze procitati, a bas
    # ona je ono sto rad tvrdi.
    fig, ax = plt.subplots(figsize=(7.5, 6.1))
    _panel(ax, city, imena, lon, lat, put, gustina, rl,
           "Mreža koju politika predlaže za Novi Sad", podnaslov(rl),
           style.color_for("RL sampling"), prvi=True, lw=2.1, pad=0.07)
    fig.tight_layout()
    # imena zona tek posle tight_layout: ono pomera osu, pa bi se izmereni
    # okviri natpisa raspali i natpisi bi se opet preklopili. Idu samo na ovoj,
    # samostalnoj karti; na dva panela jedan uz drugi pojedu trasu.
    _natpisi(ax, imena, lon, lat)
    for p in style.save(fig, REZULTATI / "novisad-rl"):
        print("->", p)

    # Kontekst uz glavnu kartu: postojeca mreza i najbolja klasicna metoda.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.9))
    for i, (ax, ime, net, boja) in enumerate([
            (axes[0], "GSP, postojeća mreža", gsp, "#1a1a1a"),
            (axes[1], "lokalna pretraga", hc, style.color_for("hill climbing"))]):
        _panel(ax, city, imena, lon, lat, put, gustina, net, ime,
               podnaslov(net), boja, prvi=(i == 0))
    fig.tight_layout()
    for p in style.save(fig, REZULTATI / "novisad-poredjenje-karta"):
        print("->", p)

    # struktura linija: brojevi koji idu u tabelu u radu, da tvrdnja o
    # "dugim petljama i patrljcima" ne ostane utisak sa slike
    redovi = ["| mreža | linija $\\geq 5$ zona | linija od 2 zone | "
              "najviše linija kroz jednu zonu |", "|---|---|---|---|"]
    for ime, net in (("GSP, postojeća", gsp), ("lokalna pretraga", hc),
                     ("politika, uzorkovanje 32", rl)):
        duzine = [len(r) for r in net.routes]
        po_zoni = np.bincount([v for r in net.routes for v in r], minlength=city.n)
        redovi.append(f"| {ime} | {sum(d >= 5 for d in duzine)} | "
                      f"{sum(d == 2 for d in duzine)} | {po_zoni.max()} |")
    (REZULTATI / "novisad-struktura.md").write_text(
        "# Struktura linija na Novom Sadu\n\n" + "\n".join(redovi) + "\n",
        encoding="utf-8")
    print("\n".join(redovi))


if __name__ == "__main__":
    main()
