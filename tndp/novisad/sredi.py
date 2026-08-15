import csv
import json
import math
import re

from tndp.novisad import konstante

# prag rastojanja za spajanje NSmart stanice sa GSP stajalistem; dve baze su nezavisno digitalizovane pa se ista tacka
PRAG_SPOJA_M = 25

# niz all_stations je nadovezivanje voznji svih varijanti linije; sece se tamo gde uzastopna stajalista nisu susedna
PRAG_SECENJA_M = 1500

TIPOVI = {"1": "gradska", "2": "prigradska", "3": "medjumesna"}


def _ucitaj(ime):
    return json.loads((konstante.RAW / ime).read_text(encoding="utf-8"))


def _upisi_csv(ime, zaglavlje, redovi):
    konstante.DATA.mkdir(parents=True, exist_ok=True)
    with open(konstante.DATA / ime, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(zaglavlje)
        w.writerows(redovi)
    print(f"  {ime}: {len(redovi)} redova")


def metara(a, b):
    return math.hypot((a[1] - b[1]) * 78000.0, (a[0] - b[0]) * 111320.0)


# GSP-ova stajalista nose tarifnu zonu koje u NSmart-u nema
def _gsp_tacke():
    tacke = {}
    for zapis in _ucitaj("gsp_stajalista.json").values():
        for red in zapis.get("stajalista") or []:
            p = red.split("|")
            if len(p) < 6:
                continue
            try:
                lon, lat = float(p[1]), float(p[2])
            except ValueError:
                continue
            tacke[(round(lon, 6), round(lat, 6))] = p[5].strip()
    return [(lat, lon, zona) for (lon, lat), zona in tacke.items()]


def stajalista():
    ns = _ucitaj("nsmart.json")
    gsp = _gsp_tacke()
    redovi = []
    bez_zone = 0
    for s in ns["stations"]:
        c = (float(s["coordinates"]["latitude"]), float(s["coordinates"]["longitude"]))
        najblizi = min(gsp, key=lambda g: metara(c, g))
        zona = najblizi[2] if metara(c, najblizi) <= PRAG_SPOJA_M else ""
        bez_zone += not zona
        redovi.append([s["id"], s.get("station_id", ""), s["name"], f"{c[0]:.7f}",
                       f"{c[1]:.7f}", s["city_name"], zona,
                       ";".join(sorted(set(s.get("lines_for_station") or [])))])
    redovi.sort(key=lambda r: r[5])
    _upisi_csv("stajalista.csv",
               ["id", "sifra", "naziv", "lat", "lon", "mesto", "zona", "linije"], redovi)
    print(f"    bez tarifne zone (nema GSP par u {PRAG_SPOJA_M} m): {bez_zone}")
    return {int(s["id"]): (float(s["coordinates"]["latitude"]),
                           float(s["coordinates"]["longitude"])) for s in ns["stations"]}


MIN_STAJALISTA = 3  # segment kraci od ovoga nije ruta nego ostatak nadovezivanja


def _segmenti(ids, koord):
    delovi, tekuci = [], [ids[0]]
    for a, b in zip(ids, ids[1:]):
        if metara(koord[a], koord[b]) > PRAG_SECENJA_M:
            delovi.append(tekuci)
            tekuci = [b]
        else:
            tekuci.append(b)
    delovi.append(tekuci)
    return sorted(delovi, key=len, reverse=True)


# svaki segment je jedna varijanta linije, ne samo najduzi: linija koja u selu ide jednosmernom petljom ima stajalista
def linije(koord):
    ns = _ucitaj("nsmart.json")
    redovi = []
    pokriveno = set()
    for linija in ns["lines"]:
        ids = [int(x) for x in linija["all_stations"] if int(x) in koord]
        if len(ids) < MIN_STAJALISTA:
            continue
        pokriveno |= set(ids)
        delovi = [d for d in _segmenti(ids, koord) if len(d) >= MIN_STAJALISTA]
        for k, ruta in enumerate(delovi):
            duzina = sum(metara(koord[x], koord[y])
                         for x, y in zip(ruta, ruta[1:])) / 1000.0
            redovi.append([linija["line_number_for_display"], k,
                           TIPOVI.get(linija["line_type"], "?"), linija["line_title"],
                           linija["direction_id_for_display"], len(ruta), f"{duzina:.2f}",
                           len(delovi), ";".join(str(x) for x in ruta)])
    redovi.sort(key=lambda r: (r[2], len(r[0]), r[0], r[1]))
    _upisi_csv("linije.csv",
               ["oznaka", "varijanta", "tip", "naziv", "smer", "stajalista",
                "duzina_km", "varijanti", "ruta"], redovi)

    na_rutama = set()
    for r in redovi:
        na_rutama |= {int(x) for x in r[8].split(";")}
    print(f"    linija: {len({r[0] for r in redovi})}, varijanti: {len(redovi)}")
    print(f"    stajališta na rutama: {len(na_rutama)} (u sirovim nizovima {len(pokriveno)}, "
          f"izgubljeno {len(pokriveno - na_rutama)})")
    puno = [r[0] for r in redovi if r[2] == "gradska" and r[7] > 6]
    if puno:
        print(f"    gradske sa >6 varijanti (proveriti ručno): {', '.join(sorted(set(puno)))}")


# polasci su u HTML tabeli: <b>SAT</b> pa niz minuta, VAR je oznaka varijante linije
def polasci():
    podaci = _ucitaj("polasci.json")
    redovi = []
    for kljuc, zapis in podaci.items():
        rezim, dan, kod = kljuc.split("|")
        for smer, td in zip("AB", re.findall(r"<td[^>]*valign=['\"]?top[^>]*>(.*?)</td>",
                                             zapis["html"], re.S)):
            sat = None
            for m in re.finditer(r"<b>(\d{2})</b>|<span class='([^']*)'>(\d{2})<b>([^<]*)</b>",
                                 td):
                if m.group(1):
                    sat = int(m.group(1))
                elif sat is not None:
                    redovi.append([konstante.REZIMI[rezim], konstante.DANI[dan], kod,
                                   zapis["naziv"], smer, f"{sat:02d}:{m.group(3)}",
                                   (m.group(4) or "").strip(),
                                   int("niskopodni" in (m.group(2) or ""))])
    _upisi_csv("polasci.csv",
               ["rezim", "dan", "linija", "naziv", "smer", "vreme", "varijanta",
                "niskopodni"], redovi)


# granice iz OSM-a dolaze kao neuredjene spoljne linije relacije
def mesne_zajednice():
    from shapely.geometry import mapping
    from shapely.ops import linemerge, polygonize, unary_union

    osm = _ucitaj("mz_osm.json")
    feats = []
    for el in osm["elements"]:
        linije = [[(t["lon"], t["lat"]) for t in cl["geometry"]]
                  for cl in el.get("members", [])
                  if cl.get("role") == "outer" and cl.get("geometry")]
        poligoni = list(polygonize(linemerge(linije))) if linije else []
        if not poligoni:
            print(f"    bez zatvorene granice: {el['tags'].get('name')}")
            continue
        geom = unary_union(poligoni)
        naziv = el["tags"].get("name", "").replace("МЗ ", "").strip()
        feats.append({"type": "Feature",
                      "properties": {"osm_id": el["id"], "naziv": naziv,
                                     "ref": el["tags"].get("ref:RS:mesna_zajednica", "")},
                      "geometry": mapping(geom)})
    (konstante.DATA / "mz.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
        encoding="utf-8")
    print(f"  mz.geojson: {len(feats)} mesnih zajednica")


def _tabela(ime):
    html = (konstante.RAW / ime).read_text(encoding="utf-8")
    redovi = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        celije = [re.sub(r"<[^>]+>", "", c).strip()
                  for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if len(celije) >= 6 and celije[0]:
            redovi.append(celije)
    return redovi[1:]  # prvi red je zaglavlje


def _broj(s):
    return int(s.replace(".", "").replace(" ", "") or 0)


def stanovnistvo():
    for ime, izlaz, kolona in [("nsinfo_mz.html", "mz_stanovnistvo.csv", "mz"),
                               ("nsinfo_naselja.html", "naselja_stanovnistvo.csv", "naselje")]:
        redovi = [[r[0], _broj(r[5]), _broj(r[4]), _broj(r[1]), _broj(r[2]), _broj(r[3])]
                  for r in _tabela(ime)]
        _upisi_csv(izlaz, [kolona, "stanovnika", "povrsina_m2", "prostora",
                           "kolektivno", "individualno"], redovi)
        print(f"    ukupno stanovnika: {sum(r[1] for r in redovi):,}".replace(",", "."))


def putnici():
    _upisi_csv("putnici_2017.csv", ["linija", "voznji_radni_dan"],
               [[k, v] for k, v in konstante.PUTNICI_2017.items()])


def main():
    print("stajališta")
    koord = stajalista()
    print("linije")
    linije(koord)
    print("polasci")
    polasci()
    print("mesne zajednice")
    mesne_zajednice()
    print("stanovništvo")
    stanovnistvo()
    print("brojanje putnika 2017")
    putnici()
    print(f"\ngotovo, sređeni podaci u {konstante.DATA}")


if __name__ == "__main__":
    main()
