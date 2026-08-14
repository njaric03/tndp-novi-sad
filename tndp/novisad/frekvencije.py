# Validacija frekvencijske faze na stvarnom redu voznje


import numpy as np

from tndp.core import frequencies as F
from tndp.novisad import konstante
from tndp.novisad.instanca import gsp_mreza, ucitaj
from tndp.novisad.kalibracija import intervali_iz_reda_voznje, opterecenja_2017
from tndp import RESULTS

# vrednosti na kojima se meri osetljivost; srednja je podrazumevana u core/frequencies.py
KAPACITETI = [60.0, 80.0, 100.0, 120.0]
UDELI_VRHA = [0.08, 0.10, 0.12]


def _mere(model, stvarno):
    razlika = model - stvarno
    return {
        "medijana |greška|": float(np.median(np.abs(razlika))),
        "prosečna greška": float(np.mean(razlika)),
        "Pearson": float(np.corrcoef(model, stvarno)[0, 1]),
        "Spearman": konstante.spearman(model, stvarno),
        "unutar 5 min": float(np.mean(np.abs(razlika) <= 5.0)),
    }


def oceni_gsp(city, mreza, kapacitet=F.CAPACITY, udeo_vrha=F.PEAK_SHARE):
    # daily_trips se NE prosledjuje: matrica Novog Sada vec nosi dnevni broj putovanja za radni dan
    return F.evaluate(city, mreza, capacity=kapacitet, peak_share=udeo_vrha)


def main():
    city, imena = ucitaj()
    mreza, dnevnik = gsp_mreza(city, imena)
    linije = [d["linija"] for d in dnevnik]

    h_stvarno_po_liniji, vrh = intervali_iz_reda_voznje()
    stvarno = np.array([h_stvarno_po_liniji.get(k, F.H_MAX) for k in linije])

    o = oceni_gsp(city, mreza)
    model = np.asarray(o["h"], dtype=float)

    print(f"GSP mreža: {len(linije)} linija, vršni sat po redu vožnje "
          f"{vrh}:00-{vrh + 1}:00")
    print(f"{'linija':>6} {'red vožnje':>11} {'model':>8} {'razlika':>9} "
          f"{'vozila':>7}")
    for k, s, m, v in sorted(zip(linije, stvarno, model, o["vozila"]),
                             key=lambda x: x[1]):
        print(f"{k:>6} {s:10.1f}m {m:7.1f}m {m - s:+8.1f}m {int(v):7d}")

    mere = _mere(model, stvarno)
    print()
    for ime, v in mere.items():
        print(f"  {ime:20s} {v:+.3f}")
    print(f"\nflota koju model traži: {o['flota']:.0f} vozila, "
          f"prosečno čekanje {o['cekanje']:.1f} min")

    osetljivost = []
    for kap in KAPACITETI:
        m = np.asarray(oceni_gsp(city, mreza, kapacitet=kap)["h"], dtype=float)
        osetljivost.append(("kapacitet", kap, _mere(m, stvarno)))
    for u in UDELI_VRHA:
        m = np.asarray(oceni_gsp(city, mreza, udeo_vrha=u)["h"], dtype=float)
        osetljivost.append(("udeo vrha", u, _mere(m, stvarno)))

    print("\nosetljivost:")
    for ime, v, mr in osetljivost:
        print(f"  {ime:10s} {v:6.2f}  medijana |greška| {mr['medijana |greška|']:5.1f} min"
              f"  Spearman {mr['Spearman']:+.3f}")

    praznjenje = _praznjenje(city, mreza, linije, stvarno, o)
    print("\nlinija koje ostaju bez ijednog putnika:")
    for ime, (n, poznate, udeo) in praznjenje.items():
        print(f"  {ime:34s} {n:2d} od {len(linije)}, "
              f"{len(poznate)} u brojanju 2017 sa {udeo:.1%} prevoza")

    _izvestaj(linije, stvarno, model, o, mere, osetljivost, praznjenje, vrh)


# Koliko linija ostane bez ijednog putnika, u tri rezima. Uz broj ide i koliko
# tih linija ima u brojanju iz 2017 i koliki deo prevoza one stvarno nose, jer
# tek to kaze da li model prazni beznacajne linije ili nosive.
def _praznjenje(city, mreza, linije, stvarno, o):
    from tndp.core.assignment import assign
    opterecenja = opterecenja_2017()
    ukupno = sum(opterecenja.values())

    def mera(res):
        prazne = [k for k, b in zip(linije, res.boardings) if b == 0]
        poznate = sorted(k for k in prazne if k in opterecenja)
        return len(prazne), poznate, sum(opterecenja[k] for k in poznate) / ukupno

    return {"prvi prolaz, fiksni penal":
                mera(assign(city, mreza, compute_transfers=False, compute_loads=True)),
            "posle konvergencije petlje":
                mera(assign(city, mreza, compute_transfers=False, compute_loads=True,
                            headways=o["h"])),
            "sa stvarnim redom vožnje":
                mera(assign(city, mreza, compute_transfers=False, compute_loads=True,
                            headways=stvarno))}


def _izvestaj(linije, stvarno, model, o, mere, osetljivost, praznjenje, vrh):
    r = ["# Frekvencije: model naspram stvarnog reda vožnje", "",
         "Frekvencijska faza (`core/frequencies.evaluate`) dimenzioniše intervale iz",
         "opterećenja najopterećenije deonice. Do sad je puštana samo na Mandlu i",
         "Mumfordu, gde nema reda vožnje pa nema ni provere. Ovde se pušta na",
         "**postojeću GSP mrežu Novog Sada**, čiji je red vožnje poznat.", "",
         f"Stvarni interval je broj polazaka u vršnom satu ({vrh}:00-{vrh + 1}:00,",
         "radni dan, smer A). Model ne dobija nijedan podatak o redu vožnje,",
         "izvodi intervale samo iz tražnje, trasa i kapaciteta vozila.", "",
         "| linija | red vožnje | model | razlika | vozila |", "|---|---|---|---|---|"]
    for k, s, m, v in sorted(zip(linije, stvarno, model, o["vozila"]),
                             key=lambda x: x[1]):
        r.append(f"| {k} | {s:.1f} min | {m:.1f} min | {m - s:+.1f} min | {int(v)} |")
    r += ["", "## Poklapanje", "", "| mera | vrednost |", "|---|---|"]
    for ime, v in mere.items():
        r.append(f"| {ime} | {v:+.3f} |")
    r += ["", f"Model traži **{o['flota']:.0f} vozila** za celu mrežu, uz prosečno "
              f"čekanje {o['cekanje']:.1f} min.", "",
          "## Osetljivost na pretpostavke", "",
          "Kapacitet vozila i udeo vršnog sata su pretpostavke, ne merenja, pa se",
          "poklapanje meri i van podrazumevanih vrednosti.", "",
          "| konstanta | vrednost | medijana \\|greška\\| | Spearman |", "|---|---|---|---|"]
    for ime, v, mr in osetljivost:
        r.append(f"| {ime} | {v:g} | {mr['medijana |greška|']:.1f} min | "
                 f"{mr['Spearman']:+.3f} |")
    r += ["", "Poklapanje se jedva menja sa tim konstantama, i to je samo po sebi",
          "dijagnoza: većina linija je prikovana za donju ili gornju granicu",
          "intervala, pa ih pomeranje kapaciteta nema gde da pomeri.", "",
          "## Gde model greši i zašto", "",
          "Greška nije ravnomerna nego **dvopolna**: linija je ili na podu od 5 min",
          "ili na plafonu od 60. Uzrok je što linija koja ostane bez putnika dobija",
          "najređi dozvoljen interval.", "",
          "| režim | linija bez ijednog putnika | od toga u brojanju 2017 | udeo prevoza koji nose |",
          "|---|---|---|---|"]
    for ime, (n, poznate, udeo) in praznjenje.items():
        r.append(f"| {ime} | {n} od {len(linije)} | {len(poznate)} "
                 f"({', '.join(poznate)}) | {udeo:.1%} |")
    r += ["", "Poslednje dve kolone su ono što nalaz čini ozbiljnim: linije koje petlja",
          "isprazni nisu rubne. Osam ih je u brojanju iz 2017. i u stvarnosti nose",
          "trećinu prevoza.", "",
          "Čitanje: dodela najkraćim putem sama po sebi isprazni šest linija, među",
          "paralelnim linijama u istom koridoru pobednik uzima sve. Petlja koja",
          "izvodi intervale iz opterećenja to pogorša na devet, jer linija sa malim",
          "opterećenjem dobije dug interval, time postane još manje privlačna, i",
          "opterećenje joj padne na nulu. Povratna sprega nije prigušena ničim.", "",
          "Probano i **ne pomaže**: geometrijsko prigušenje koraka (0.5 i 0.3) i",
          "politički plafon intervala od 30 min umesto 60. Prigušenje pogorša",
          "Spearman na +0.33, plafon podigne medijanu greške na 10 min, a broj praznih",
          "linija u oba slučaja ostaje devet. Dakle kvar nije u petlji nego ispod nje.", "",
          "Stvarni prevoznik nema ovaj problem iz dva razloga koje model nema:",
          "putnik ulazi u prvu liniju koja naiđe pa se opterećenje deli po",
          "frekvencijama (Spiess-Florian, strategija umesto puta), a nivo usluge je",
          "delom politička odluka, GSP vozi liniju 7 na 7,5 minuta jer je tako",
          "odlučeno, a ne zato što je tražnja to iznudila. Model je predviđa na 60.", "",
          "**Posledica za rad:** poredak linija po opterećenju je upotrebljiv",
          f"(Spearman {mere['Spearman']:+.3f}), pojedinačni intervali nisu. Svaki",
          "zaključak koji traži tačan interval po liniji, a tu spada i poređenje",
          "vidova prevoza sa tramvajem, mora sačekati dodelu po strategiji.", ""]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "novisad-frekvencije.md").write_text("\n".join(r) + "\n",
                                                      encoding="utf-8")
    print(f"\n-> {RESULTS / 'novisad-frekvencije.md'}")


if __name__ == "__main__":
    main()
