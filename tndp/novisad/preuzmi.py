import json
import re
import time

import requests

from tndp.novisad import konstante

PAUZA = 0.2  # sekundi između zahteva, da se sajt ne gnjavi


def _sesija():
    s = requests.Session()
    s.headers["User-Agent"] = "tndp-seminarski/0.1 (akademska upotreba)"
    s.verify = False
    return s


def _upisi(ime, sadrzaj):
    konstante.RAW.mkdir(parents=True, exist_ok=True)
    put = konstante.RAW / ime
    if isinstance(sadrzaj, (dict, list)):
        put.write_text(json.dumps(sadrzaj, ensure_ascii=False), encoding="utf-8")
    else:
        put.write_text(sadrzaj, encoding="utf-8")
    print(f"  {ime}  {put.stat().st_size // 1024} KB")


# odgovori GSP-a nisu čist JSON nego imaju HTML whitespace oko niza, a u samom
# HTML-u pre niza zna da bude uglasta zagrada, pa se traži prva koja stvarno
# otvara niz stringova
def _json_iz_html(tekst):
    kraj = tekst.rfind("]")
    for m in re.finditer(r"\[", tekst):
        if m.start() > kraj:
            break
        try:
            return json.loads(tekst[m.start():kraj + 1])
        except json.JSONDecodeError:
            continue
    return []


# katalog linija sa stranice mreže: numerički id, kategorija, relacija, oznaka
def linije_sa_mreze(html):
    pat = re.compile(
        r'id="(\d+)"[^>]*class="button-linija (grad|prigrad|medjumesni) ablin\d+"'
        r'[^>]*title="([^"]*)"[^>]*>\s*([^<]+?)\s*</a>')
    return [{"id": int(a), "kategorija": b, "relacija": c.strip(), "oznaka": d.strip()}
            for a, b, c, d in pat.findall(html)]


# NSmart ugrađuje celu bazu stanica i linija u izvor stranice kao JS string
# literal g_in_cities; vadi se ručnim skeniranjem jer regex preko 1.9 MB
# escapeovanog JSON-a nije pouzdan
def izvuci_nsmart(html):
    i = html.find("var g_in_cities")
    if i < 0:
        raise ValueError("g_in_cities nije nađen u izvoru stranice")
    j = html.find('"', i)
    k = j + 1
    while k < len(html):
        if html[k] == "\\":
            k += 2
            continue
        if html[k] == '"':
            break
        k += 1
    return json.loads(json.loads(html[j:k + 1]))


def preuzmi_nsmart(s):
    print("NSmart portal")
    html = s.get(f"{konstante.NSMART}/sr/prikaz-svih-linija", timeout=60).text
    if "g_in_cities" not in html:
        html = s.get(f"{konstante.NSMART}/", timeout=60).text
    podaci = izvuci_nsmart(html)
    print(f"  stanica {len(podaci['stations'])}, linija {len(podaci['lines'])}, "
          f"mesta {len(podaci['cities'])}")
    _upisi("nsmart.json", podaci)


def preuzmi_gsp_mrezu(s):
    print("GSP mreža linija")
    html = s.get(f"{konstante.GSP}/mreza", timeout=40).text
    _upisi("mreza.html", html)
    linije = linije_sa_mreze(html)
    print(f"  linija u katalogu: {len(linije)}")

    stajalista, trase = {}, {}
    for br, ln in enumerate(linije):
        for ime, url, kutija in [("stajalista", "mreza-get-stajalista-tacke", stajalista),
                                 ("trasa", "mreza-get-linija-tacke", trase)]:
            r = s.get(f"{konstante.GSP}/{url}", params={"linija": ln["id"]}, timeout=30)
            kutija[ln["oznaka"]] = {"meta": ln, ime: _json_iz_html(r.text)}
            time.sleep(PAUZA)
        if br % 50 == 0:
            print(f"  ... {br}/{len(linije)}")
    _upisi("gsp_stajalista.json", stajalista)
    _upisi("gsp_trase.json", trase)


def preuzmi_zone(s):
    print("tarifne zone")
    zone = {}
    for i in range(1, 7):
        zone[i] = s.get(f"{konstante.GSP}/cenovnik-sen/zona-mesta/{i}", timeout=30).text
        time.sleep(PAUZA)
    _upisi("zona_mesta.json", zone)


def preuzmi_polaske(s):
    print("red vožnje")
    polasci = {}
    for rv in konstante.REZIMI:
        for dan in konstante.DANI:
            r = s.get(f"{konstante.GSP}/red-voznje/lista-linija",
                      params={"rv": rv, "vaziod": konstante.VAZIOD, "dan": dan}, timeout=30)
            kodovi = re.findall(r'<option value="([^"]+)">([^<]*)</option>', r.text)
            for kod, naziv in kodovi:
                r2 = s.get(f"{konstante.GSP}/red-voznje/ispis-polazaka",
                           params={"rv": rv, "vaziod": konstante.VAZIOD, "dan": dan,
                                   "linija[]": kod}, timeout=30)
                polasci[f"{rv}|{dan}|{kod}"] = {"naziv": naziv.strip(), "html": r2.text}
                time.sleep(PAUZA)
            print(f"  {konstante.REZIMI[rv]} / {konstante.DANI[dan]}: {len(kodovi)} linija")
    _upisi("polasci.json", polasci)


# Overpass zna da vrati 504 kad je opterećen, pa se pokušava više puta
def preuzmi_mesne_zajednice(s, pokusaja=5):
    print("mesne zajednice iz OSM-a")
    j, i, k, a = konstante.BBOX
    upit = (f'[out:json][timeout:140];rel["admin_level"="10"]({j},{i},{k},{a});'
            f'out geom 300;')
    for n in range(pokusaja):
        r = s.post(konstante.OVERPASS, data={"data": upit}, timeout=200)
        try:
            podaci = r.json()
        except ValueError:
            print(f"  pokušaj {n + 1}: HTTP {r.status_code}, čekam")
            time.sleep(20)
            continue
        print(f"  relacija: {len(podaci['elements'])}")
        _upisi("mz_osm.json", podaci)
        return
    raise RuntimeError("Overpass nije odgovorio ni posle više pokušaja")


def preuzmi_nsinfo(s):
    print("stanovništvo (JKP Informatika)")
    for ime, put in [("nsinfo_mz.html", "broj-stanovnika-po-mesnim-zajednicama"),
                     ("nsinfo_naselja.html", "broj-stanovnika-po-naseljima")]:
        _upisi(ime, s.get(f"{konstante.NSINFO}/{put}", timeout=40).text)
        time.sleep(PAUZA)


# osmnx podrazumevano keširа u ./cache; sklanja se pod data/novisad/ koji je
# ionako u .gitignore
def _podesi_osmnx():
    import osmnx as ox

    ox.settings.cache_folder = str(konstante.RAW / "osmnx_cache")


# ulična mreža za vremena vožnje; graf je MultiDiGraph pa dvosmerna ulica daje
# dve usmerene grane a jednosmerna jednu, što je poželjno za najkraće puteve
def preuzmi_ulice():
    import osmnx as ox

    _podesi_osmnx()
    print("ulična mreža (OSM)")
    konstante.RAW.mkdir(parents=True, exist_ok=True)
    put = konstante.RAW / "ulice.graphml"
    if put.exists():
        print(f"  {put.name} već postoji, preskačem")
        return
    g = ox.graph_from_bbox(konstante.BBOX_ULICE, network_type="drive")
    g = ox.add_edge_travel_times(ox.add_edge_speeds(g))
    ox.save_graphml(g, put)
    print(f"  čvorova {g.number_of_nodes()}, grana {g.number_of_edges()}, "
          f"{put.stat().st_size // 1024} KB")


# sadržaji za stranu privlačnosti gravitacionog modela
def preuzmi_sadrzaje():
    import osmnx as ox

    _podesi_osmnx()
    print("sadržaji (OSM)")
    put = konstante.RAW / "sadrzaji.geojson"
    if put.exists():
        print(f"  {put.name} već postoji, preskačem")
        return
    g = ox.features_from_bbox(konstante.BBOX_ULICE, {t: True for t in konstante.POI_TAGOVI})
    g = g.copy()
    g["geometry"] = g.geometry.representative_point()  # poligoni -> tačke
    kolone = ["geometry"] + [t for t in konstante.POI_TAGOVI if t in g.columns]
    g = g[kolone].reset_index(drop=True)
    g.to_file(put, driver="GeoJSON")
    print(f"  tačaka {len(g)}, {put.stat().st_size // 1024} KB")


def main():
    import urllib3
    urllib3.disable_warnings()
    s = _sesija()
    preuzmi_nsmart(s)
    preuzmi_gsp_mrezu(s)
    preuzmi_zone(s)
    preuzmi_polaske(s)
    preuzmi_mesne_zajednice(s)
    preuzmi_nsinfo(s)
    preuzmi_ulice()
    preuzmi_sadrzaje()
    print(f"\ngotovo, sirovi podaci u {konstante.RAW}")


if __name__ == "__main__":
    main()
