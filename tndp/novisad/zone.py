import csv
import json
import unicodedata

from tndp.novisad import izvori

# područje studije je ono što opslužuje GRADSKI saobraćaj, a ne tarifna zona I.
# zona I je zona u kojoj važi gradska karta i šira je: od njenih 603 stajališta
# samo 59% ima gradsku liniju, a Sremska Kamenica, Rumenka, Bukovac, Ledinci,
# Šangaj i Paragovo nemaju nijednu. GSP i sam vodi Službu gradskog i Službu
# prigradskog saobraćaja kao odvojene celine (informator o radu), a TNDP
# redizajnira skup linija koje bi planer redizajnirao zajedno.
TIP_LINIJE = "gradska"
MIN_STAJALISTA_U_ZONI = 1

CIRILICA = "АБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШ"
LATINICA = ["A", "B", "V", "G", "D", "Đ", "E", "Ž", "Z", "I", "J", "K", "L", "LJ",
            "M", "N", "NJ", "O", "P", "R", "S", "T", "Ć", "U", "F", "H", "C", "Č",
            "DŽ", "Š"]

# nazivi u OSM-u i u evidenciji JKP Informatika se ponegde razilaze: dve su
# štamparske greške u OSM-u, jedna je varijanta imena
ALIJASI = {"LEDNICI": "LEDINCI", "OMALDINSKIPOKRET": "OMLADINSKIPOKRET",
           "PEJICEVISALASI": "PEJICEVISALASINEMANOVCI"}


def _kljuc(naziv):
    prevod = dict(zip(CIRILICA, LATINICA))
    s = "".join(prevod.get(ch, ch) for ch in naziv.upper())
    s = s.replace("III", "3").replace("II", "2")
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = "".join(c for c in s if c.isalnum())
    return ALIJASI.get(s, s)


def _ucitaj_mz():
    from shapely.geometry import shape

    fc = json.loads((izvori.DATA / "mz.geojson").read_text(encoding="utf-8"))
    return [(f["properties"]["naziv"], f["properties"]["ref"], shape(f["geometry"]))
            for f in fc["features"]]


def _ucitaj_stajalista():
    with open(izvori.DATA / "stajalista.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# stajališta kroz koja prolazi bar jedna linija traženog tipa
def _stajalista_tipa(tip=TIP_LINIJE):
    with open(izvori.DATA / "linije.csv", encoding="utf-8") as f:
        na_rutama = set()
        for r in csv.DictReader(f):
            if r["tip"] == tip:
                na_rutama |= set(r["ruta"].split(";"))
    return na_rutama


def _ucitaj_stanovnistvo():
    with open(izvori.DATA / "mz_stanovnistvo.csv", encoding="utf-8") as f:
        return {_kljuc(r["mz"]): (int(r["stanovnika"]), int(r["povrsina_m2"]))
                for r in csv.DictReader(f)}


# nivoi iz evidencije JKP Informatika broje prijavljene a ne stanovnike i za
# Grad daju 414.789 naspram popisnih 368.967; udeli po MZ se zadržavaju, a
# nivoi se spuštaju faktorom popis/evidencija računatim po naselju
def _faktori_popisa():
    with open(izvori.DATA / "naselja_stanovnistvo.csv", encoding="utf-8") as f:
        evidencija = {_kljuc(r["naselje"]): int(r["stanovnika"])
                      for r in csv.DictReader(f)}
    return {k: izvori.POPIS_2022[n] / evidencija[k]
            for n in izvori.POPIS_2022 if (k := _kljuc(n)) in evidencija}


def izgradi():
    from shapely.geometry import Point

    mz = _ucitaj_mz()
    stajalista = _ucitaj_stajalista()
    stanovnistvo = _ucitaj_stanovnistvo()
    faktori = _faktori_popisa()
    gradska = _stajalista_tipa()

    # svako stajalište pada u najviše jednu mesnu zajednicu
    pripadnost = {}
    van = 0
    for r in stajalista:
        tacka = Point(float(r["lon"]), float(r["lat"]))
        pogodak = next((n for n, _, g in mz if g.contains(tacka)), None)
        if pogodak is None:
            van += 1
            continue
        pripadnost.setdefault(pogodak, []).append(r)
    print(f"stajališta van svih mesnih zajednica: {van}/{len(stajalista)}")

    redovi = []
    for naziv, ref, geom in mz:
        svoja = pripadnost.get(naziv, [])
        zona1 = [r for r in svoja if r["zona"] == "I"]
        na_gradskoj = [r for r in svoja if r["id"] in gradska]
        u_studiji = len(na_gradskoj) >= MIN_STAJALISTA_U_ZONI

        # reprezentativna tačka zone je težište njenih stajališta na gradskim
        # linijama, a ne geometrijsko težište poligona: velike rubne MZ su
        # uglavnom njive, a putovanja nastaju tamo gde su stajališta
        osnov = na_gradskoj or zona1 or svoja
        if osnov:
            lon = sum(float(r["lon"]) for r in osnov) / len(osnov)
            lat = sum(float(r["lat"]) for r in osnov) / len(osnov)
        else:
            lon, lat = geom.centroid.x, geom.centroid.y

        k = _kljuc(naziv)
        sirovo, povrsina = stanovnistvo.get(k, (0, 0))
        # faktor naselja kome MZ pripada; MZ unutar grada nose faktor naselja Novi Sad
        faktor = faktori.get(k, faktori.get(_kljuc("NOVI SAD"), 1.0))
        redovi.append([naziv, ref, int(round(sirovo * faktor)), sirovo, povrsina,
                       f"{geom.area * 78 * 111.32:.2f}", f"{lat:.6f}", f"{lon:.6f}",
                       len(svoja), len(zona1), len(na_gradskoj), int(u_studiji)])

    redovi.sort(key=lambda r: (-r[11], -r[2]))
    izvori.DATA.mkdir(parents=True, exist_ok=True)
    with open(izvori.DATA / "zone.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mz", "ref", "stanovnika", "stanovnika_evidencija", "povrsina_m2",
                    "povrsina_km2", "lat", "lon", "stajalista", "stajalista_zona1",
                    "stajalista_gradska", "u_studiji"])
        w.writerows(redovi)

    # pripadnost stajališta zoni je potrebna svakom koraku posle ovoga (prevođenje
    # GSP ruta u nizove zona), a računa se samo ovde — zato se i upisuje
    u_zoni = {r[0] for r in redovi if r[11]}
    veza = sorted(((s["id"], naziv, int(naziv in u_zoni))
                   for naziv, svoja in pripadnost.items() for s in svoja),
                  key=lambda x: (-x[2], x[1]))
    with open(izvori.DATA / "stajalista_zone.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stajaliste_id", "mz", "u_studiji"])
        w.writerows(veza)
    print(f"stajalista_zone.csv: {len(veza)} veza, "
          f"{sum(v[2] for v in veza)} u području studije")

    u_studiji = [r for r in redovi if r[11]]
    print(f"zone.csv: {len(redovi)} mesnih zajednica, {len(u_studiji)} u području studije")
    print(f"  stanovnika u području studije: {sum(r[2] for r in u_studiji):,}".replace(",", "."))
    bez = [r[0] for r in u_studiji if r[2] == 0]
    if bez:
        print(f"  bez podatka o stanovništvu: {', '.join(bez)}")
    return redovi


def main():
    redovi = izgradi()
    print(f"\n{'mesna zajednica':30s} {'stan.':>8s} {'km2':>7s} {'staj.':>6s} "
          f"{'zona I':>7s} {'gradska':>8s}")
    for r in redovi:
        if not r[11]:
            continue
        print(f"{r[0]:30s} {r[2]:8d} {r[5]:>7s} {r[8]:6d} {r[9]:7d} {r[10]:8d}")
    print("\nvan područja studije (nema gradsku liniju):")
    for r in redovi:
        if not r[11]:
            print(f"  {r[0]:28s} stajališta {r[8]:3d}, u zoni I {r[9]:3d}")


if __name__ == "__main__":
    main()
