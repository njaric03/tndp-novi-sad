import csv

import numpy as np

from tndp.novisad import konstante
from tndp.novisad.ulice import ucitaj_zone

# eksponent opadanja sa daljinom. Kalibracija na opterećenja linija iz 2017.
# (tndp/novisad/kalibracija.py, results/novisad-kalibracija.md) NE razlikuje
# betu: greška poklapanja je ista na tri decimale za svaku betu u [0.5, 2.0].
# Zadržava se 2.0, ista vrednost koju koristi synth.py — tako se
# raspodela tražnje na Novom Sadu ne razlikuje od one na kojoj je politika
# trenirana, iz istog razloga zbog kog featuri idu kroz rang transformaciju.
BETA = 2.0

# donja granica rastojanja, ista kao u generatoru: sprečava da bliske zone dobiju
# beskonačnu tražnju
MIN_KM = 0.3

# ispod ovog rastojanja putovanje ne ulazi u gradski prevoz. Ovo JESTE
# kalibrisano — greška poklapanja ima jasan minimum na 3.5 km, i bez tog
# parametra beta uopšte nije određena (bez praga kalibracija betu tera na
# nulu, dakle na model u kom daljina ne igra ulogu).
#
# 3.5 km je previše za pešačenje i ne treba ga tako čitati: to je granica
# ispod koje autobus gubi od pešačenja, bicikla i automobila zajedno. Novi Sad
# je ravan i biciklistički grad, a zone su guste, pa je granica visoka.
PESACKI_PRAG = 3.5


def _rastojanja(zone):
    lat = np.array([float(r["lat"]) for r in zone])
    lon = np.array([float(r["lon"]) for r in zone])
    dy = (lat[:, None] - lat[None, :]) * 111.32
    dx = (lon[:, None] - lon[None, :]) * 78.0
    return np.hypot(dx, dy)


def _tau(zone):
    with open(konstante.DATA / "tau.csv", encoding="utf-8") as f:
        redovi = list(csv.reader(f))
    zaglavlje = redovi[0][1:]
    poredak = [zaglavlje.index(r["mz"]) for r in zone]
    m = np.array([[float(x) for x in red[1:]] for red in redovi[1:]])
    return m[np.ix_(poredak, poredak)]


def _privlacnost(zone):
    with open(konstante.DATA / "privlacnost.csv", encoding="utf-8") as f:
        po_zoni = {r["mz"]: float(r["privlacnost"]) for r in csv.DictReader(f)}
    return np.array([po_zoni[r["mz"]] for r in zone])


# gravitacioni model, ista formula kao u synth.py da se trening i test
# raspodela ne bi razlikovale: produkcija puta privlačnost kroz rastojanje na
# beta, pa simetrizacija i skaliranje na ukupan broj putovanja. razlika je što
# su ovde produkcija i privlačnost mereni, a ne izvučeni iz lognormalne, i što
# nema multiplikativnog šuma.
def izgradi(beta=BETA, mera="euklidsko", ukupno=None, prag=PESACKI_PRAG):
    zone = ucitaj_zone()
    imena = [r["mz"] for r in zone]
    n = len(zone)
    prod = np.array([float(r["stanovnika"]) for r in zone])
    attr = _privlacnost(zone)
    ukupno = ukupno or konstante.TREND_PUTNIKA[2017]

    euklid = _rastojanja(zone)
    d = euklid if mera == "euklidsko" else _tau(zone)
    d = np.maximum(d, MIN_KM if mera == "euklidsko" else 1.0)

    traznja = prod[:, None] * attr[None, :] / d ** beta
    traznja = (traznja + traznja.T) / 2.0
    np.fill_diagonal(traznja, 0.0)
    # prag se meri euklidski bez obzira na to kojom merom opada tražnja: reč je
    # o tome koliko je daleko, ne koliko se dugo vozi
    if prag > 0:
        traznja = np.where(euklid < prag, 0.0, traznja)
    traznja *= ukupno / traznja.sum()
    return imena, traznja


def _upisi(imena, traznja):
    konstante.DATA.mkdir(parents=True, exist_ok=True)
    with open(konstante.DATA / "traznja.csv", "w", encoding="utf-8", newline="") as f:
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

    print(f"traznja.csv: {n}x{n}, ukupno {traznja.sum():,.0f} putovanja".replace(",", "."))
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
    print(f"\nprag od {PESACKI_PRAG} km je kalibrisan na opterećenja linija iz 2017.\n"
          "(results/novisad-kalibracija.md); bez njega je pri beta=2.0 čak 81% tražnje\n"
          "padalo na parove kraće od 2 km, što za autobusko putovanje nije uverljivo.\n"
          "BETA se kalibracijom NE razlikuje — greška poklapanja je ista na tri\n"
          "decimale za svaku betu — pa ostaje 2.0, koliko koristi i sintetički\n"
          "generator, da se trening i test raspodela ne bi razlikovale.")


if __name__ == "__main__":
    main()
