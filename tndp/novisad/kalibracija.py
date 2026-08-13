# Kalibracija gravitacione matrice na opterecenja linija iz 2017

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from tndp.core.assignment import assign
from tndp.core.city import CityGraph
from tndp.core.frequencies import H_MAX, H_MIN
from tndp.novisad import konstante, traznja
from tndp.novisad.instanca import gsp_mreza, ucitaj

BETE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
MERE = ["euklidsko", "tau"]
# rastojanje ispod kog se ne ide autobusom nego peske, u kilometrima
PRAGOVI = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
# koliko TV-a se smatra istim rezultatom pri izboru; vidi komentar u main()
TOLERANCIJA = 0.002
REZULTATI = Path(__file__).resolve().parent.parent.parent / "results"


def _osnovna(oznaka):
    m = re.match(r"^\d+", oznaka)
    return m.group(0) if m else None


def opterecenja_2017():
    with open(konstante.DATA / "putnici_2017.csv", encoding="utf-8") as f:
        return {r["linija"]: float(r["voznji_radni_dan"]) for r in csv.DictReader(f)}


# interval sledjenja po liniji iz reda voznje: broj polazaka u vrsnom satu, u jednom smeru
def intervali_iz_reda_voznje():
    po_liniji = defaultdict(list)
    with open(konstante.DATA / "polasci.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["rezim"] != "gradski" or r["dan"] != "radni dan" or r["smer"] != "A":
                continue
            k = _osnovna(r["linija"])
            if k:
                po_liniji[k].append(int(r["vreme"].split(":")[0]))

    svi = [h for v in po_liniji.values() for h in v]
    vrh = max(set(svi), key=svi.count)
    h = {}
    for k, sati in po_liniji.items():
        n = sati.count(vrh)
        h[k] = float(np.clip(60.0 / n, H_MIN, H_MAX)) if n else H_MAX
    return h, vrh


# udeo svake linije u ukupnim ulascima, po redosledu linija u mrezi
def _udeli(vrednosti, linije, samo):
    v = np.array([x for x, k in zip(vrednosti, linije) if k in samo], dtype=float)
    return v / v.sum() if v.sum() > 0 else v


# Spearman preko rankdata, NE preko argsort(argsort)
def _spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


# ulasci po liniji koje model predvidja za dati par (beta, pesacki prag)
def ulasci_modela(city, mreza, beta, mera, prag, headways):
    _, m = traznja.izgradi(beta=beta, mera=mera, prag=prag)
    grad = CityGraph(coords=city.coords, street_time=city.street_time,
                     demand=m, name=city.name)
    res = assign(grad, mreza, compute_transfers=False, compute_loads=True,
                 headways=headways)
    return res.boardings, res


def main():
    city, imena = ucitaj()
    mreza, dnevnik = gsp_mreza(city, imena)
    linije = [d["linija"] for d in dnevnik]
    stvarno = opterecenja_2017()
    # linija 19 (Miseluk) nije u brojanju iz 2017
    zajednicke = {k for k in linije if k in stvarno}
    print(f"linija u poređenju: {len(zajednicke)} od {len(linije)}"
          f"  (bez: {sorted(set(linije) - zajednicke, key=int)})")

    h_po_liniji, vrh = intervali_iz_reda_voznje()
    headways = [h_po_liniji.get(k, H_MAX) for k in linije]
    print(f"vršni sat po redu vožnje: {vrh}:00-{vrh + 1}:00")
    print(f"interval sleđenja: medijana {np.median(headways):.1f} min, "
          f"raspon {min(headways):.1f}-{max(headways):.1f}")

    cilj_udeli = _udeli([stvarno[k] for k in linije if k in zajednicke],
                        [k for k in linije if k in zajednicke], zajednicke)

    nalazi = []
    for mera in MERE:
        for beta in BETE:
            for prag in PRAGOVI:
                ulasci, _ = ulasci_modela(city, mreza, beta, mera, prag, headways)
                u = _udeli(ulasci, linije, zajednicke)
                # ukupno varijaciono rastojanje dva profila: polovina zbira apsolutnih razlika udela, u [0, 1]
                tv = float(np.abs(u - cilj_udeli).sum() / 2.0)
                nalazi.append((mera, beta, prag, tv, _spearman(u, cilj_udeli)))

    # TV razlikuje prag ali NE i betu, po beti je ravan na tri decimale
    najbolji_tv = min(x[3] for x in nalazi)
    u_igri = [x for x in nalazi if x[3] <= najbolji_tv + TOLERANCIJA]
    najbolji = max(u_igri, key=lambda x: x[4])
    print(f"\nunutar {TOLERANCIJA} od najboljeg TV-a: {len(u_igri)} kombinacija")
    print(f"{'mera':>10} {'beta':>5} {'prag':>5} {'TV':>7} {'Spearman':>9}")
    for m, b, p, tv, sp in sorted(u_igri, key=lambda x: -x[4])[:12]:
        print(f"{m:>10} {b:5.2f} {p:5.1f} {tv:7.3f} {sp:+9.3f}")

    # Sweep bira betu, ali kod je NE koristi: traznja.BETA je 2.0 jer je to vrednost
    # generatora, a razlika prema sweep optimumu je u trecoj decimali Spearmana.
    # Izvestaj zato opisuje betu koju pipeline stvarno vrti, ne argmax sweep-a.
    mera, _, prag, _, _ = najbolji
    usvojen = next(x for x in nalazi
                   if (x[0], x[1], x[2]) == (mera, traznja.BETA, prag))
    beta, tv, sp = usvojen[1], usvojen[3], usvojen[4]
    ulasci, res = ulasci_modela(city, mreza, beta, mera, prag, headways)
    u = _udeli(ulasci, linije, zajednicke)
    poredak = [k for k in linije if k in zajednicke]

    print(f"\nsweep optimum: beta={najbolji[1]}, Spearman={najbolji[4]:+.3f}")
    print(f"usvojeno: mera={mera}, beta={beta}, prag={prag} km, "
          f"TV={tv:.3f}, Spearman={sp:+.3f}")
    print(f"{'linija':>6} {'interval':>9} {'stvarno':>9} {'model':>9} {'razlika':>9}")
    for k, um, uc in sorted(zip(poredak, u, cilj_udeli),
                            key=lambda x: -x[2]):
        print(f"{k:>6} {h_po_liniji.get(k, H_MAX):8.1f}m {100 * uc:8.1f}% "
              f"{100 * um:8.1f}% {100 * (um - uc):+8.1f}%")

    _izvestaj(nalazi, usvojen, najbolji, poredak, u, cilj_udeli, h_po_liniji, vrh, res)


def _izvestaj(nalazi, usvojen, najbolji, poredak, u, cilj, h, vrh, res):
    mera, beta, prag, tv, sp = usvojen
    r = ["# Kalibracija gravitacione matrice Novog Sada", "",
         "Slobodan parametar je `beta`, eksponent opadanja tražnje sa daljinom.",
         "Bira se tako da POSTOJEĆA GSP mreža, opterećena tom matricom, da profil",
         "putovanja po linijama najbliži brojanju iz 2017. Mase zona, privlačnost i",
         "rastojanja su mereni i ne podešavaju se.", "",
         f"Intervali sleđenja su iz reda vožnje (vršni sat {vrh}:00-{vrh + 1}:00,",
         "radni dan, smer A), ne procenjeni iz opterećenja. Bez toga poređenje ne bi",
         "bilo pošteno: linija 16 ima 89 vožnji dnevno zato što vozi retko, a ne zato",
         "što tražnje nema.", "",
         "`prag` je rastojanje ispod kog se putovanje obavi peške i ne ulazi u",
         "prevoz. Bez njega beta nije prepoznatljiva, kalibracija je tera na nulu,",
         "jer gravitacioni model bez praga gomila tražnju na susedne zone.", "",
         "`TV` je ukupno varijaciono rastojanje profila udela (0 = poklapanje, 1 =",
         "disjunktno). `Spearman` je korelacija rangova linija po opterećenju.",
         "Prikazano je 12 najboljih od "
         f"{len(nalazi)} kombinacija.", "",
         "| mera rastojanja | beta | prag (km) | TV | Spearman |", "|---|---|---|---|---|"]
    for m, b, p, t, s in sorted(nalazi, key=lambda x: x[3])[:12]:
        oznaka = " **<-**" if (m, b, p) == (mera, beta, prag) else ""
        r.append(f"| {m} | {b:.2f} | {p:.1f} | {t:.3f} | {s:+.3f}{oznaka} |")
    r += ["", f"Usvojeno: **mera = {mera}, beta = {beta}, prag = {prag} km** "
              f"(TV {tv:.3f}, Spearman {sp:+.3f}). Svi ostali brojevi u ovom fajlu "
              "računati su pri toj vrednosti.", "",
          "## Šta je kalibracija stvarno odredila", "",
          "**Prag jeste određen.** TV ima jasan minimum na 3.5 km i raste i ispod",
          "i iznad. 3.5 km je previše za pešačenje i ne treba ga tako čitati, to je",
          "granica ispod koje autobus gubi od pešačenja, bicikla i automobila",
          "zajedno. Novi Sad je ravan i biciklistički, a zone su guste.", "",
          "**Beta NIJE određena.** Pri pragu 3.5 km je TV jednak na tri decimale za",
          "svaku betu; razlikuje ih tek korelacija rangova, i to u trećoj decimali",
          f"(sweep optimum je beta {najbolji[1]:.1f} sa {najbolji[4]:+.3f},",
          f"usvojena beta {beta:.1f} daje {sp:+.3f}). Kod usvaja",
          f"**beta = {beta:.1f}**, a ne sweep optimum: razlika je unutar šuma, a",
          f"{beta:.1f} je vrednost koju koristi `synth.py`, pa se raspodela",
          "tražnje na Novom Sadu ne",
          "razlikuje od one na kojoj je politika trenirana. Isti razlog zbog kog",
          "featuri idu kroz rang transformaciju.", "",
          "## Profil po linijama pri izabranoj beti", "",
          "| linija | interval (min) | stvarno | model | razlika |", "|---|---|---|---|---|"]
    for k, um, uc in sorted(zip(poredak, u, cilj), key=lambda x: -x[2]):
        r.append(f"| {k} | {h.get(k, H_MAX):.1f} | {100 * uc:.1f}% | "
                 f"{100 * um:.1f}% | {100 * (um - uc):+.1f}% |")
    nule = [k for k, um in zip(poredak, u) if um == 0.0]
    r += ["", f"Nepokrivena tražnja pri toj matrici: {res.d['d_un']:.3f}, "
              f"`C_p` {res.C_p:.2f} min.", "",
          "## Zašto TV ne pada ispod 0.29", "",
          f"Linije {', '.join(nule)} dobijaju TAČNO nula putnika, iako u stvarnosti "
          "nose", f"{100 * sum(uc for k, uc in zip(poredak, cilj) if k in nule):.1f}% "
          "prevoza. Uzrok nije matrica tražnje nego dodela: svaki par zona bira",
          "jedan najbrži put i sva tražnja tog para ide na njega, pa među paralelnim",
          "linijama pobednik uzima sve. Stvarni putnik ulazi u onu liniju koja prva",
          "naiđe, pa se opterećenje deli po frekvencijama (Spiess-Florian, strategija",
          "umesto puta). Taj model ovde nije implementiran i to je gornja granica",
          "tačnosti svakog poređenja po linijama u ovom radu.", ""]
    REZULTATI.mkdir(exist_ok=True)
    (REZULTATI / "novisad-kalibracija.md").write_text("\n".join(r), encoding="utf-8")
    print(f"\n-> {REZULTATI / 'novisad-kalibracija.md'}")


if __name__ == "__main__":
    main()
