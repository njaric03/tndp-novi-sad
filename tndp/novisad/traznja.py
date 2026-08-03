import csv

import numpy as np

from tndp.novisad import izvori
from tndp.novisad.ulice import ucitaj_zone

# eksponent opadanja sa daljinom. NIJE kalibrisan na opterećenja linija —
# ta kalibracija se ne može uraditi, razlog je u results/novisad_kalibracija.md
# (jedanaest od 33 linije dobija nula putnika pri svakoj beti, jer dodela ide
# najkraćim putem pa jedna linija u koridoru uzme sve). Vrednost je izabrana po
# uverljivosti prosečne dužine putovanja: pri 2.0 je ona 1.42 km i 81% tražnje
# pada na parove kraće od 2 km, što nije autobusko putovanje; pri 0.5 je 2.56 km.
BETA = 0.5

# zbir matrice. Brojanje iz 2017 daje 172.687 ULAZAKA, a matrica nosi PUTOVANJA
# — putnik koji presedne broji se dvaput. Odnos meren na GSP mreži pri BETA je
# 1.289, pa je zbir 172687 / 1.289. Ranije je ovde stajalo samih 172.687, što je
# precenjivalo tražnju za taj faktor.
PUTOVANJA = 133_947

# donja granica rastojanja, ista kao u generatoru: sprečava da bliske zone dobiju
# beskonačnu tražnju
MIN_KM = 0.3


def _rastojanja(zone):
    lat = np.array([float(r["lat"]) for r in zone])
    lon = np.array([float(r["lon"]) for r in zone])
    dy = (lat[:, None] - lat[None, :]) * 111.32
    dx = (lon[:, None] - lon[None, :]) * 78.0
    return np.hypot(dx, dy)


def _tau(zone):
    with open(izvori.DATA / "tau.csv", encoding="utf-8") as f:
        redovi = list(csv.reader(f))
    zaglavlje = redovi[0][1:]
    poredak = [zaglavlje.index(r["mz"]) for r in zone]
    m = np.array([[float(x) for x in red[1:]] for red in redovi[1:]])
    return m[np.ix_(poredak, poredak)]


def _privlacnost(zone):
    with open(izvori.DATA / "privlacnost.csv", encoding="utf-8") as f:
        po_zoni = {r["mz"]: float(r["privlacnost"]) for r in csv.DictReader(f)}
    return np.array([po_zoni[r["mz"]] for r in zone])


# gravitacioni model, ista formula kao u synth/generator.py da se trening i test
# raspodela ne bi razlikovale: produkcija puta privlačnost kroz rastojanje na
# beta, pa simetrizacija i skaliranje na ukupan broj putovanja. razlika je što
# su ovde produkcija i privlačnost mereni, a ne izvučeni iz lognormalne, i što
# nema multiplikativnog šuma.
def izgradi(beta=BETA, mera="euklidsko", ukupno=None):
    zone = ucitaj_zone()
    imena = [r["mz"] for r in zone]
    n = len(zone)
    prod = np.array([float(r["stanovnika"]) for r in zone])
    attr = _privlacnost(zone)
    ukupno = ukupno or PUTOVANJA

    d = _rastojanja(zone) if mera == "euklidsko" else _tau(zone)
    d = np.maximum(d, MIN_KM if mera == "euklidsko" else 1.0)

    traznja = prod[:, None] * attr[None, :] / d ** beta
    traznja = (traznja + traznja.T) / 2.0
    np.fill_diagonal(traznja, 0.0)
    traznja *= ukupno / traznja.sum()
    return imena, traznja


def _upisi(imena, traznja):
    izvori.DATA.mkdir(parents=True, exist_ok=True)
    with open(izvori.DATA / "traznja.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mz"] + imena)
        for i, ime in enumerate(imena):
            w.writerow([ime] + [f"{x:.1f}" for x in traznja[i]])


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    imena, traznja = izgradi()
    _upisi(imena, traznja)
    n = len(imena)
    gore = np.triu_indices(n, 1)

    ukupno_str = f"{traznja.sum():,.0f}".replace(",", ".")
    print(f"traznja.csv: {n}x{n}, ukupno {ukupno_str} putovanja")
    print(f"  parova sa tražnjom > 100: {(traznja[gore] > 100).sum()} od {len(gore[0])}")

    zbir = traznja.sum(axis=1)
    poredak = np.argsort(-zbir)
    print("\nnajjače zone po ukupnoj tražnji (dolazak + polazak):")
    for i in poredak[:8]:
        print(f"  {imena[i]:26s} {zbir[i]:8.0f}  ({100 * zbir[i] / traznja.sum():.1f}%)")

    print("\nnajjači parovi:")
    par = sorted(((traznja[i, j], imena[i], imena[j]) for i, j in zip(*gore)), reverse=True)
    for v, a, b in par[:8]:
        print(f"  {a:24s} - {b:24s} {v:7.0f}")

    d = _rastojanja(ucitaj_zone())
    print(f"\nrastojanja između zona: medijana {np.median(d[gore]):.2f} km, "
          f"maksimum {d[gore].max():.2f} km")

    _, alt = izgradi(mera="tau")
    print(f"Spearman euklidsko rastojanje vs vreme vožnje: "
          f"{_spearman(traznja[gore], alt[gore]):+.3f}")

    print("\nosetljivost na beta (prosek je ponderisan tražnjom):")
    print(f"{'beta':>5s} {'prosečna dužina':>16s} {'udeo < 2 km':>12s} {'udeo > 5 km':>12s}")
    for b in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        _, m = izgradi(beta=b)
        w = m[gore]
        print(f"{b:5.1f} {(w * d[gore]).sum() / w.sum():13.2f} km "
              f"{100 * w[d[gore] < 2].sum() / w.sum():11.0f}% "
              f"{100 * w[d[gore] > 5].sum() / w.sum():11.0f}%")
    print(f"\nBETA = {BETA} je izabrana po uverljivosti prosečne dužine putovanja, NE\n"
          "kalibrisana na opterećenja linija — ta kalibracija nije moguća sa dodelom\n"
          "bez frekvencija (11 od 33 linije dobija nula putnika pri svakoj beti).\n"
          "Detalji i tabela: results/novisad_kalibracija.md, pravi se sa\n"
          "python -m tndp.novisad.kalibracija")


if __name__ == "__main__":
    main()
