# Podloga za karte: ulicna mreza i granice zona, iz lokalnih fajlova.
# graphml se cita direktno preko ElementTree, bez osmnx, da crtanje ne povlaci
# ceo geo stek kao zavisnost.

import json
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np

from tndp.novisad import konstante

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
# hijerarhija ulica koja se crta deblje; ostalo je tanka mreza
MAGISTRALNE = {"motorway", "trunk", "primary", "motorway_link", "trunk_link",
               "primary_link"}


def _kljucevi(koren):
    m = {}
    for k in koren.findall("g:key", NS):
        m[(k.get("for"), k.get("attr.name"))] = k.get("id")
    return m


# vraca (obicne, magistralne): svaka je lista segmenata [(lon0, lat0), (lon1, lat1)]
@lru_cache(maxsize=1)
def ulice():
    koren = ET.parse(konstante.RAW / "ulice.graphml").getroot()
    k = _kljucevi(koren)
    graf = koren.find("g:graph", NS)

    xy = {}
    for cvor in graf.findall("g:node", NS):
        d = {e.get("key"): e.text for e in cvor}
        xy[cvor.get("id")] = (float(d[k[("node", "x")]]), float(d[k[("node", "y")]]))

    obicne, glavne = [], []
    kljuc_hw = k[("edge", "highway")]
    for ivica in graf.findall("g:edge", NS):
        a, b = ivica.get("source"), ivica.get("target")
        if a not in xy or b not in xy:
            continue
        d = {e.get("key"): e.text for e in ivica}
        hw = (d.get(kljuc_hw) or "").strip("[]'\" ").split(",")[0]
        (glavne if hw in MAGISTRALNE else obicne).append((xy[a], xy[b]))
    return obicne, glavne


# Vodene povrsine, pre svega Dunav. Crta se samo zato sto rad na reku racuna:
# Petrovaradin je najjaca zona jer se preko Dunava ne ide peske, a tvrdnja da
# mreze prelaze reku samo tamo gde most postoji bez reke se ne moze proveriti.
# U proracun ne ulazi nigde; traznja zna za euklidsko rastojanje, a ulicni graf
# za mostove, jer su oni ulice kao i sve druge.
@lru_cache(maxsize=1)
def vode():
    put = konstante.RAW / "vode.geojson"
    if not put.exists():  # karta se crta i bez reke, samo losije
        return []
    fc = json.loads(put.read_text(encoding="utf-8"))
    out = []
    for f in fc["features"]:
        g = f.get("geometry") or {}
        if g.get("type") == "Polygon":
            delovi = g["coordinates"]
        elif g.get("type") == "MultiPolygon":
            delovi = [d for mp in g["coordinates"] for d in mp]
        else:
            continue
        # samo spoljni prsten: ostrva u reci su na ovoj razmeri sitna, a
        # rupe u poligonu bi tražile crtanje preko Path-a sa kodovima
        out.append(np.array(delovi[0], dtype=float))
    return out


# povrsina prstena po formuli trapeza; sluzi samo da se medju vodama nadje
# najveca, a to je Dunav
def _povrsina(pr):
    x, y = pr[:, 0], pr[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


# Natpis na samoj reci, kurzivom kako se hidronimi i pisu na kartama. Bez njega
# siva povrsina moze da se procita kao zona van studije, a postoji i mesna
# zajednica Dunav, pa je "Dunav" na karti dosad znacilo nju.
#
# Ide posle set_xlim/set_ylim, kao i razmernik: mesto se trazi u vidljivom
# prozoru, jer se Dunav pruza i van kadra pa bi natpis inace izasao iz slike.
# Centar okvira kod krivudave reke padne na obalu, zato se trazi najduza
# vodoravna tetiva unutar prozora i uzima njena sredina.
def natpis_vode(ax, x0, x1, y0, y1, ime="Dunav"):
    import matplotlib.patheffects as pe
    from matplotlib.path import Path

    povrsine = vode()
    if not povrsine:
        return
    put = Path(max(povrsine, key=_povrsina))
    # margina, da natpis ne legne uz sam rub kadra
    mx, my = (x1 - x0) * 0.06, (y1 - y0) * 0.06
    najbolje, najduza = None, 0.0
    for y in np.linspace(y0 + my, y1 - my, 60):
        unutra = [x for x in np.linspace(x0 + mx, x1 - mx, 200)
                  if put.contains_point((x, y))]
        if len(unutra) > 1 and unutra[-1] - unutra[0] > najduza:
            najduza = unutra[-1] - unutra[0]
            najbolje = ((unutra[0] + unutra[-1]) / 2, y)
    if najbolje is None:
        return
    ax.text(*najbolje, ime, fontsize=8, style="italic", color="#41545c",
            ha="center", va="center", zorder=2,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="#d3dadd")])


# granice po zoni: {ime: [prsten, ...]}, samo zone u studiji
@lru_cache(maxsize=1)
def granice_po_zoni(imena):
    fc = json.loads((konstante.DATA / "mz.geojson").read_text(encoding="utf-8"))
    trazene = set(imena)
    out = {}
    for f in fc["features"]:
        ime = f["properties"]["naziv"]
        if ime not in trazene:
            continue
        g = f["geometry"]
        delovi = g["coordinates"] if g["type"] == "Polygon" else             [d for mp in g["coordinates"] for d in mp]
        out.setdefault(ime, []).extend(np.array(pr, dtype=float) for pr in delovi)
    return out


# spoljne granice zona iz mz.geojson, samo one koje su u studiji
@lru_cache(maxsize=1)
def granice(imena):
    fc = json.loads((konstante.DATA / "mz.geojson").read_text(encoding="utf-8"))
    trazene = set(imena)
    out = []
    for f in fc["features"]:
        if f["properties"]["naziv"] not in trazene:
            continue
        g = f["geometry"]
        delovi = g["coordinates"] if g["type"] == "Polygon" else \
            [d for mp in g["coordinates"] for d in mp]
        for prsten in delovi:
            out.append(np.array(prsten, dtype=float))
    return out


# Zone se boje isecenim delom Blues-a. Pun raspon bi najsvetliju zonu obojio
# belo, pa se ne bi razlikovala od podloge van studije, a najtamniju tako tamno
# da natpis preko nje vise ne moze da se procita.
@lru_cache(maxsize=1)
def skala():
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "zone", colormaps["Blues"](np.linspace(0.12, 0.74, 256)))


# nacrtaj podlogu na osu; vraca ScalarMappable za legendu boje, ili None ako
# zone nisu obojene vrednoscu
def nacrtaj(ax, imena, vrednosti=None):
    from matplotlib.collections import LineCollection

    # voda ide ispod svega: gde se zona i reka preklapaju, zona pobedjuje, jer
    # je ona predmet studije. Ton je siv a ne iz Blues skale kojom se boje zone,
    # inace bi se reka procitala kao zona sa najvecom traznjom
    # Ton je namerno bez plavog: zone se boje Blues skalom, pa je Petrovaradin,
    # kao najjaca zona, tamnoplav i vec izgleda kao reka. Voda zato ide sivo,
    # sa izrazenim obodom koji prati obalu, da se skala i podloga ne mesaju
    for pr in vode():
        ax.fill(pr[:, 0], pr[:, 1], color="#d3dadd", zorder=-1, linewidth=1.1,
                edgecolor="#7e929b")
    obicne, glavne = ulice()
    ax.add_collection(LineCollection(obicne, linewidths=0.35, colors="#c8c8c8",
                                     alpha=0.7, zorder=0))
    ax.add_collection(LineCollection(glavne, linewidths=1.1, colors="#9a9a9a",
                                     alpha=0.95, zorder=0))
    # zona se boji svojom vrednoscu; traznja je svojstvo zone, ne tacke u njoj
    po_zoni = granice_po_zoni(tuple(imena))
    sm = None
    if vrednosti is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=float(np.min(vrednosti)), vmax=float(np.max(vrednosti)))
        cmap = skala()
        for ime, v in zip(imena, vrednosti):
            for pr in po_zoni.get(ime, []):
                ax.fill(pr[:, 0], pr[:, 1], color=cmap(norm(v)),
                        alpha=0.9, zorder=0, linewidth=0)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.asarray(vrednosti))
    else:
        for prsteni in po_zoni.values():
            for pr in prsteni:
                ax.fill(pr[:, 0], pr[:, 1], color="#f2f2f2", alpha=0.55, zorder=0)
    for p in granice(tuple(imena)):
        ax.plot(p[:, 0], p[:, 1], lw=0.6, color="#8f8f8f", alpha=0.85, zorder=1)
    return sm


# razmernik u kilometrima, dole levo
def razmernik(ax, lon0, lon1, lat0, km=2.0):
    du = km / (111.32 * np.cos(np.radians(lat0)))
    x0 = lon0 + (lon1 - lon0) * 0.04
    y0 = lat0
    ax.plot([x0, x0 + du], [y0, y0], lw=2.2, color="#333333", zorder=6,
            solid_capstyle="butt")
    ax.text(x0 + du / 2, y0 + (lon1 - lon0) * 0.004, f"{km:g} km", ha="center",
            va="bottom", fontsize=7, color="#333333", zorder=6)


# Trase koje prate stvarne ulice umesto pravih linija izmedju tezista zona.
# Prava linija implicira ulicu koje nema; ovde se svaki skok izmedju susednih
# zona zameni najkracim putem kroz OSM graf, pa trasa lezi na kolovozu.
@lru_cache(maxsize=1)
def _graf():
    from scipy.sparse import csr_matrix

    koren = ET.parse(konstante.RAW / "ulice.graphml").getroot()
    k = _kljucevi(koren)
    graf = koren.find("g:graph", NS)

    ids, xy = {}, []
    for cvor in graf.findall("g:node", NS):
        d = {e.get("key"): e.text for e in cvor}
        ids[cvor.get("id")] = len(xy)
        xy.append((float(d[k[("node", "x")]]), float(d[k[("node", "y")]])))

    kl = k[("edge", "length")]
    r, c, w = [], [], []
    for ivica in graf.findall("g:edge", NS):
        a, b = ivica.get("source"), ivica.get("target")
        if a not in ids or b not in ids:
            continue
        d = {e.get("key"): e.text for e in ivica}
        duz = float(d.get(kl) or 1.0)
        r += [ids[a], ids[b]]
        c += [ids[b], ids[a]]
        w += [duz, duz]
    n = len(xy)
    return np.array(xy), csr_matrix((np.array(w), (np.array(r), np.array(c))), shape=(n, n))


# indeks najblizeg ulicnog cvora za svako teziste zone
def _snap(xy, lon, lat):
    out = []
    for x, y in zip(lon, lat):
        d = (xy[:, 0] - x) ** 2 + ((xy[:, 1] - y) * 1.4) ** 2
        out.append(int(np.argmin(d)))
    return np.array(out)


# vrati funkciju koja za par zona daje polilini ju po ulicama
def trase(lon, lat):
    from scipy.sparse.csgraph import dijkstra

    xy, g = _graf()
    izvori = _snap(xy, lon, lat)
    _, pred = dijkstra(g, directed=False, indices=izvori, return_predecessors=True)

    def put(a, b):
        red, c = [], izvori[b]
        koren = izvori[a]
        while c != koren and c >= 0:
            red.append(c)
            c = pred[a, c]
        if c < 0:
            return np.array([[lon[a], lat[a]], [lon[b], lat[b]]])
        red.append(koren)
        return xy[red[::-1]]

    return put
