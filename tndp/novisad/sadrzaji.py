import csv
import json

from tndp.novisad import konstante

# kategorije preslikane iz njaric03/mu-novi-sad-tipologija-zgrada; redosled je bitan jer svaka tačka pada u prvu
GROCERY = {"supermarket", "convenience", "greengrocer", "bakery", "butcher", "deli"}
FOOD = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream"}
EDU = {"school", "university", "college", "kindergarten", "childcare"}
HEALTH = {"hospital", "clinic", "doctors", "pharmacy", "dentist", "veterinary"}
CIVIC = {"townhall", "police", "courthouse", "community_centre", "library",
         "place_of_worship", "fire_station", "post_office"}


def kategorija(shop, amenity, office, leisure):
    if shop in GROCERY:
        return "grocery"
    if amenity in FOOD:
        return "food"
    if amenity in EDU:
        return "education"
    if amenity in HEALTH:
        return "health"
    if amenity in CIVIC:
        return "civic"
    if amenity == "bank" or office:
        return "office"
    if leisure:
        return "leisure"
    if shop:
        return "retail"
    return None


def izgradi():
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    from tndp.novisad.ulice import ucitaj_zone

    zone = ucitaj_zone()
    imena = [r["mz"] for r in zone]
    fc = json.loads((konstante.DATA / "mz.geojson").read_text(encoding="utf-8"))
    geom = {f["properties"]["naziv"]: shape(f["geometry"]) for f in fc["features"]}
    poligoni = [geom[ime] for ime in imena]
    stablo = STRtree(poligoni)

    sirovo = json.loads((konstante.RAW / "sadrzaji.geojson").read_text(encoding="utf-8"))
    kategorije = list(konstante.TEZINE_SADRZAJA)
    broj = {ime: dict.fromkeys(kategorije, 0) for ime in imena}
    bez_kategorije = van_zona = 0

    for f in sirovo["features"]:
        p = f["properties"]
        kat = kategorija(p.get("shop") or "", p.get("amenity") or "",
                         p.get("office") or "", p.get("leisure") or "")
        if kat is None:
            bez_kategorije += 1
            continue
        tacka = Point(*f["geometry"]["coordinates"])
        pogodak = next((i for i in stablo.query(tacka) if poligoni[i].contains(tacka)), None)
        if pogodak is None:
            van_zona += 1
            continue
        broj[imena[pogodak]][kat] += 1

    ukupno = sum(sum(v.values()) for v in broj.values())
    print(f"sadržaja: {len(sirovo['features'])} ukupno, {ukupno} razvrstano u zone, "
          f"{bez_kategorije} bez kategorije, {van_zona} van područja studije")

    redovi = []
    for ime in imena:
        b = broj[ime]
        tezinski = sum(konstante.TEZINE_SADRZAJA[k] * b[k] for k in kategorije)
        redovi.append([ime] + [b[k] for k in kategorije] + [sum(b.values()),
                                                            f"{tezinski:.1f}"])
    redovi.sort(key=lambda r: -float(r[-1]))
    with open(konstante.DATA / "privlacnost.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mz"] + kategorije + ["ukupno", "privlacnost"])
        w.writerows(redovi)
    print(f"privlacnost.csv: {len(redovi)} zona")

    po_kategoriji = {k: sum(broj[i][k] for i in imena) for k in kategorije}
    print("  po kategoriji:", po_kategoriji)
    prazne = [r[0] for r in redovi if r[-2] == 0]
    if prazne:
        print(f"  zone bez ijednog sadržaja: {prazne}")
    return redovi


def main():
    redovi = izgradi()
    print(f"\n{'mesna zajednica':30s} {'ukupno':>7s} {'težinski':>9s}  najjače kategorije")
    kategorije = list(konstante.TEZINE_SADRZAJA)
    for r in redovi[:15]:
        top = sorted(zip(kategorije, r[1:1 + len(kategorije)]), key=lambda x: -x[1])[:3]
        print(f"{r[0]:30s} {r[-2]:7d} {r[-1]:>9s}  "
              + ", ".join(f"{k} {v}" for k, v in top if v))


if __name__ == "__main__":
    main()
