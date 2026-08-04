import csv

import numpy as np

from tndp.novisad import konstante


def ucitaj_zone(samo_studija=True):
    with open(konstante.DATA / "zone.csv", encoding="utf-8") as f:
        redovi = [r for r in csv.DictReader(f)
                  if not samo_studija or r["u_studiji"] == "1"]
    return redovi


# vremena vožnje između zona po stvarnoj uličnoj mreži; graf je usmeren pa
# jednosmerne ulice daju tau[i][j] != tau[j][i], a CityGraph traži simetriju,
# te se na kraju uzima prosek dva smera
def izgradi(prag_snap_m=500):
    import networkx as nx
    import osmnx as ox

    zone = ucitaj_zone()
    g = ox.load_graphml(konstante.RAW / "ulice.graphml")
    print(f"ulična mreža: {g.number_of_nodes()} čvorova, {g.number_of_edges()} grana")

    lat = np.array([float(r["lat"]) for r in zone])
    lon = np.array([float(r["lon"]) for r in zone])
    cvorovi = ox.nearest_nodes(g, lon, lat)
    udaljenost = [ox.distance.great_circle(la, lo, g.nodes[c]["y"], g.nodes[c]["x"])
                  for la, lo, c in zip(lat, lon, cvorovi)]
    daleko = [(zone[i]["mz"], round(d)) for i, d in enumerate(udaljenost) if d > prag_snap_m]
    print(f"težište zone do najbližeg čvora mreže: medijana {np.median(udaljenost):.0f} m, "
          f"maksimum {max(udaljenost):.0f} m")
    if daleko:
        print(f"  dalje od {prag_snap_m} m: {daleko}")

    n = len(zone)
    tau = np.full((n, n), np.inf)
    for i, izvor in enumerate(cvorovi):
        duzine = nx.single_source_dijkstra_path_length(g, izvor, weight="travel_time")
        for j, cilj in enumerate(cvorovi):
            if cilj in duzine:
                tau[i, j] = duzine[cilj] / 60.0  # sekunde -> minuti
    np.fill_diagonal(tau, 0.0)

    nedostizni = int(np.isinf(tau).sum())
    if nedostizni:
        parovi = [(zone[i]["mz"], zone[j]["mz"])
                  for i, j in zip(*np.where(np.isinf(tau)))][:5]
        print(f"  nedostiznih parova: {nedostizni}, npr {parovi}")

    asimetrija = np.abs(tau - tau.T)[np.isfinite(tau) & np.isfinite(tau.T)]
    print(f"asimetrija smerova: medijana {np.median(asimetrija):.2f} min, "
          f"maksimum {asimetrija.max():.2f} min")
    tau = (tau + tau.T) / 2.0

    with open(konstante.DATA / "tau.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mz"] + [r["mz"] for r in zone])
        for i, r in enumerate(zone):
            w.writerow([r["mz"]] + [f"{x:.2f}" for x in tau[i]])
    print(f"tau.csv: {n}x{n} minuta, prosek {tau[tau > 0].mean():.1f}, "
          f"maksimum {tau.max():.1f}")
    return zone, tau


# dve zone su susedne ako im se granice dodiruju; ovo je kandidat za ulične
# ivice CityGraph-a, za razliku od tau koji je najkraći put kroz celu mrežu
def susedstvo():
    import json

    from shapely.geometry import shape

    zone = {r["mz"] for r in ucitaj_zone()}
    fc = json.loads((konstante.DATA / "mz.geojson").read_text(encoding="utf-8"))
    geom = {f["properties"]["naziv"]: shape(f["geometry"]) for f in fc["features"]
            if f["properties"]["naziv"] in zone}
    imena = sorted(geom)

    parovi = [(a, b) for i, a in enumerate(imena) for b in imena[i + 1:]
              if geom[a].buffer(1e-6).intersects(geom[b])]
    with open(konstante.DATA / "susedstvo.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b"])
        w.writerows(parovi)

    stepen = {ime: 0 for ime in imena}
    for a, b in parovi:
        stepen[a] += 1
        stepen[b] += 1
    print(f"susedstvo.csv: {len(parovi)} parova, prosečan stepen "
          f"{2 * len(parovi) / len(imena):.1f}, minimum {min(stepen.values())}")
    izolovane = [k for k, v in stepen.items() if v == 0]
    if izolovane:
        print(f"  bez suseda: {izolovane}")
    return parovi


def main():
    izgradi()
    susedstvo()


if __name__ == "__main__":
    main()
