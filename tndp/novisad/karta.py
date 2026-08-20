# Mreze linija nacrtane preko stvarne ulicne mreze Novog Sada
# podloga je OSM graf ulica i granice mesnih zajednica, vidi novisad/podloga.py
# pokretanje: python -m tndp.novisad.karta

from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D

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
# isti run iz kog su i sve tabele o Novom Sadu u radu; novisad-r19 je stariji,
# na stopi ucenja 1e-4, i njegovi brojevi se ne smeju naci na istoj strani
MODEL = KOREN / "runs" / "novisad-r19h" / "best.pt"


# tezista zona se citaju iz zone.csv, ne rekonstruisu iz centriranih koordinata
def _tezista(zone):
    return (np.array([float(z["lon"]) for z in zone]),
            np.array([float(z["lat"]) for z in zone]))


FONT_NATPIS = 8.0

# Imena zona se kroz ceo projekat drze cirilicom, jer se tako spajaju sa
# mz.geojson i ostalim tabelama. Rad je na latinici, pa se preslovljava tek pri
# crtanju: kljucevi ostaju netaknuti, menja se samo ono sto se vidi.
PRESLOVI = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E",
    "Ж": "Ž", "З": "Z", "И": "I", "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj",
    "М": "M", "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R", "С": "S",
    "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č",
    "Џ": "Dž", "Ш": "Š",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
    "џ": "dž", "ш": "š",
}

# Ime mesne zajednice je pogresno napisano u samom OpenStreetMap-u, pa greska
# prolazi kroz mz.geojson i sve izvedene tabele. Ne ispravlja se u podacima:
# kljuc mora da ostane isti kao u izvoru da bi spajanje radilo i posle ponovnog
# preuzimanja. Menja se samo natpis na karti.
ISPRAVKE = {"Омалдински покрет": "Омладински покрет"}


def _latinica(ime):
    ime = ISPRAVKE.get(ime, ime)
    return "".join(PRESLOVI.get(z, z) for z in ime)

# Kandidati za mesto natpisa u odnosu na tacku zone, redom po pozeljnosti.
# Iznad tacke je najcitljivije, pa strane, pa udaljeniji redovi.
MESTA = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2),
         (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 3), (0, -3), (2, 0), (-2, 0)]


# 32 imena se sudaraju oko centra. Probamo svako mesto iz MESTA, merimo stvarni
# okvir teksta (broj slova ne radi, nisu iste sirine) i uzimamo najmanje preklapanje
def _natpisi(ax, imena, lon, lat):
    fig = ax.figure
    fig.canvas.draw()
    render = fig.canvas.get_renderer()
    korak = FONT_NATPIS * 1.35
    imena = [_latinica(i) for i in imena]

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


# "1 linija", "3 linije", "14 linija"
def _linija(c):
    d = c % 10
    return f"{c} {'linije' if d in (2, 3, 4) and c not in (12, 13, 14) else 'linija'}"


# koliko linija koristi svaku deonicu izmedju dve susedne zone; smer se ne
# razlikuje, jer linija vozi u oba
def _opterecenje(net):
    br = {}
    for r in net.routes:
        for a, b in zip(r, r[1:]):
            k = (a, b) if a < b else (b, a)
            br[k] = br.get(k, 0) + 1
    return br


def _panel(ax, city, imena, lon, lat, put, udeo, net, naslov, podnaslov, boja,
           prvi=False, lw=1.6, pad=0.16, teret_max=None, legenda=True):
    # zone su obojene UDELOM u ukupnoj traznji, to jest istom velicinom koja se
    # navodi u tekstu; gustina po km2 je centar cinila najtamnijim iako periferne
    # zone nose veci deo traznje, pa je karta govorila suprotno od brojeva
    sm = podloga.nacrtaj(ax, imena, vrednosti=udeo)

    # Debljina deonice je broj linija koje njome prolaze, a ne jedna linija po
    # trasi. Devetnaest trasa iste boje sa sitnim pomerajem daje jednu masnu
    # crtu kroz centar iz koje se ne vidi koliko ih tuda zapravo ide, a bas to
    # je tvrdnja koju karta treba da nosi: GSP je radijalan, model nije.
    teret = _opterecenje(net)
    # skala je zajednicka za sve panele jedne figure: da je po panelu, ista
    # debljina bi levo znacila jedanaest linija a desno dve
    najveci = teret_max or max(teret.values(), default=1)

    def debljina(c):
        return lw * (0.5 + 1.7 * (c - 1) / max(najveci - 1, 1))

    # tanke prve, da najopterecenije deonice ostanu na vrhu
    for (a, b), c in sorted(teret.items(), key=lambda kv: kv[1]):
        xy = put(a, b)
        # trasa prati ulice: prava linija izmedju tezista implicira ulicu koje nema
        w = debljina(c)
        ax.plot(xy[:, 0], xy[:, 1], lw=w + 1.6, color="white", alpha=0.85,
                zorder=3, solid_capstyle="round", solid_joinstyle="round")
        ax.plot(xy[:, 0], xy[:, 1], lw=w, color=boja, alpha=0.92,
                zorder=4, solid_capstyle="round", solid_joinstyle="round")

    # Legenda debljine. Ako nijedna deonica ne nosi vise od jedne linije, sve
    # su iste debljine i legenda od jednog reda samo zbunjuje; tu cinjenicu
    # onda nosi potpis figure.
    if legenda and najveci > 1:
        nivoi = sorted({1, (1 + najveci) // 2, najveci})
        ax.legend(
            handles=[Line2D([], [], color="#4d4d4d", lw=debljina(c),
                            label=_linija(c)) for c in nivoi],
            loc="upper left", fontsize=8, handlelength=2.4, labelspacing=0.8,
            title="linija po deonici", title_fontsize=8, frameon=True,
            framealpha=0.85, edgecolor="none", borderpad=0.6)

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
    # naslov samo gde razlikuje panele. Na samostalnoj karti bi doslovno
    # ponovio potpis figure u radu, pa se prazan naslov preskace
    if naslov:
        ax.set_title(naslov, fontsize=13, pad=8, weight="bold")
    ax.text(0.5, -0.035, podnaslov, transform=ax.transAxes, ha="center",
            va="top", fontsize=10.5, color="#444444")
    podloga.natpis_vode(ax, x0, x1, y0, y1)
    if prvi:
        podloga.razmernik(ax, x0, x1, y0 + (y1 - y0) * 0.03)
    return sm


# Traka za boju zone. Potpis figure objasnjava sta plavo znaci, ali bez skale
# citalac ne moze da proceni koliko je razlika izmedju dve zone velika.
#
# Traka ide kao inset u prazan ugao same karte, ne kao colorbar uz osu: uz osu
# joj matplotlib oduzme pojas sirine cele figure, pa se sudari sa naslovom
# panela, a kartu suzi taman toliko da natpisi zona vise ne stanu.
def _traka_udela(ax, sm, mesto=(0.55, 0.02, 0.41, 0.022)):
    import matplotlib.pyplot as plt

    cax = ax.inset_axes(mesto)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    # bez natpisa: sta boja znaci pise u potpisu figure u radu, pa bi natpis na
    # samoj traci bio isto receno dvaput. Ostaju brojevi, sa procentom na traci.
    cb.ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    cb.ax.tick_params(labelsize=9.0, colors="#333333", length=2, pad=1.5)
    cb.outline.set_visible(False)
    return cb


def main(checkpoint=MODEL):
    import matplotlib.pyplot as plt

    city, imena = ucitaj()
    zone = ucitaj_zone()
    lon, lat = _tezista(zone)
    put = podloga.trase(lon, lat)
    udeo = 100.0 * city.demand.sum(1) / city.demand.sum()
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
        # decimalni zarez i tri decimale kao u tabeli rada; na dve je ista
        # vrednost stajala kao 1,769 u tabeli a 1,77 ispod slike
        return (f"cilj {style.broj(objective(res, scales, 0.5), 3)}    "
                f"$C_p$ {style.broj(res.C_p, 1)} min    "
                f"$C_o$ {res.C_o:.0f} min")

    # legenda ide u potpis figure, ne na sliku - sirok tekst bi pokvario
    # bbox_inches="tight" i karta bi u radu ispala mala
    style.apply_style()

    # Glavni rezultat studije slucaja ide sam i veliki, sa imenima zona.
    # Na tri panela jedan uz drugi se mreza politike ne moze procitati, a bas
    # ona je ono sto rad tvrdi.
    fig, ax = plt.subplots(figsize=(7.5, 6.1))
    sm = _panel(ax, city, imena, lon, lat, put, udeo, rl,
                "", podnaslov(rl),
                style.color_for("RL sampling"), prvi=True, lw=2.1, pad=0.07)
    fig.tight_layout()
    # traka tek posle tight_layout: inset se racuna od konacnog polozaja ose
    _traka_udela(ax, sm)
    # imena zona tek posle tight_layout: ono pomera osu, pa bi se izmereni
    # okviri natpisa raspali i natpisi bi se opet preklopili. Idu samo na ovoj,
    # samostalnoj karti; na dva panela jedan uz drugi pojedu trasu.
    _natpisi(ax, imena, lon, lat)
    for p in style.save(fig, REZULTATI / "novisad-rl"):
        print("->", p)

    # Kontekst uz glavnu kartu: postojeca mreza i najbolja klasicna metoda.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.9))
    paneli = [(axes[0], "GSP, postojeća mreža", gsp, "#1a1a1a"),
              (axes[1], "lokalna pretraga", hc, style.color_for("hill climbing"))]
    # obe mreze se crtaju po istoj skali tereta, inace bi tanka crta levo i
    # tanka crta desno znacile razlicit broj linija
    zajednicki = max(max(_opterecenje(n).values(), default=1) for _, _, n, _ in paneli)
    for i, (ax, ime, net, boja) in enumerate(paneli):
        sm = _panel(ax, city, imena, lon, lat, put, udeo, net, ime,
                    podnaslov(net), boja, prvi=(i == 0),
                    teret_max=zajednicki, legenda=(i == 0))
    fig.tight_layout()
    # traznja je svojstvo grada, ista je levo i desno, pa ide jedna traka
    _traka_udela(axes[0], sm, mesto=(0.54, 0.035, 0.43, 0.022))
    for p in style.save(fig, REZULTATI / "novisad-poredjenje-karta"):
        print("->", p)

    # struktura linija: brojevi koji idu u tabelu u radu, da tvrdnja o
    # "dugim petljama i patrljcima" ne ostane utisak sa slike
    # bez LaTeX-a u tabeli: results/*.md se citaju i kao obican markdown
    mreze = (("GSP, postojeća", gsp), ("lokalna pretraga", hc),
             ("politika, uzorkovanje 32", rl))
    redovi = ["| mreža | linija ≥ 5 zona | linija od 2 zone | najduža linija "
              "(zona) | najviše linija kroz jednu zonu |",
              "|---|---|---|---|---|"]
    for ime, net in mreze:
        duzine = [len(r) for r in net.routes]
        po_zoni = np.bincount([v for r in net.routes for v in r], minlength=city.n)
        redovi.append(f"| {ime} | {sum(d >= 5 for d in duzine)} | "
                      f"{sum(d == 2 for d in duzine)} | {max(duzine)} | "
                      f"{po_zoni.max()} |")

    # Koje su to zone: rad imenuje Prvu vojvodjansku brigadu i Petrovaradin, pa
    # ti brojevi moraju da se pregenerisu zajedno sa mrezom, a ne da se prepisu
    # iz ranije verzije.
    redovi += ["", "## Kroz koje zone prolazi najviše linija", "",
               "| mreža | zone sa najviše linija |", "|---|---|"]
    for ime, net in mreze:
        po_zoni = np.bincount([v for r in net.routes for v in r], minlength=city.n)
        vrh = np.argsort(po_zoni)[::-1][:4]
        redovi.append(f"| {ime} | " + ", ".join(
            f"{_latinica(imena[z])} {po_zoni[z]}" for z in vrh) + " |")

    (REZULTATI / "novisad-struktura.md").write_text(
        "# Struktura linija na Novom Sadu\n\n" + "\n".join(redovi) + "\n",
        encoding="utf-8")
    print("\n".join(redovi))


if __name__ == "__main__":
    main()
