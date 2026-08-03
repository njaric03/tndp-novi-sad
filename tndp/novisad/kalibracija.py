import csv
from pathlib import Path

import numpy as np

from tndp.core.assignment import assign
from tndp.novisad import grad as G
from tndp.novisad import izvori

# Namera je bila da se `beta` podesi na 18 izmerenih opterećenja linija iz
# 2017. Ne može — razlog je u izlazu ove skripte i u results/novisad_kalibracija.md.
# Ostaje ono što se pošteno može uraditi: izbor bete po dužini putovanja i
# ispravka ukupnog obima (brojanje meri ULASKE, matrica nosi PUTOVANJA).
BETE = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)

# prosečna dužina putovanja gradskim autobusom; ispod ovoga putovanje pripada
# pešačenju a ne prevozu. Novi Sad je kompaktan, ali 1.4 km prosečno nije
# autobusko putovanje ni u kompaktnom gradu.
UVERLJIVA_DUZINA_KM = (2.5, 5.0)


def _merenja():
    with open(izvori.DATA / "putnici_2017.csv", encoding="utf-8") as f:
        return {r["linija"]: float(r["voznji_radni_dan"]) for r in csv.DictReader(f)}


def _po_liniji(oznake, boardings):
    po = {}
    for o, b in zip(oznake, boardings):
        po[G.osnovna(o)] = po.get(G.osnovna(o), 0.0) + b
    return po


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def profil(beta, meren):
    g, imena = G.izgradi(beta=beta)
    net, oznake, odbacene = G.gsp_mreza(g, imena)
    res = assign(g, net, compute_loads=True)

    d = np.linalg.norm(g.coords[:, None, :] - g.coords[None, :, :], axis=2)
    gore = np.triu_indices(g.n, 1)
    w = g.demand[gore]

    po = _po_liniji(oznake, res.boardings)
    zajednicke = [k for k in meren if k in po]
    p = np.array([po[k] for k in zajednicke])
    m = np.array([meren[k] for k in zajednicke])

    return {
        "beta": beta,
        "duzina": float((w * d[gore]).sum() / w.sum()),
        "kratka": float(w[d[gore] < 2.0].sum() / w.sum()),
        "odnos": float(res.boardings.sum() / g.demand.sum()),
        "praznih": int((res.boardings == 0).sum()),
        "linija": len(net.routes),
        "rho": _spearman(p, m),
        "d_0": res.d["d_0"],
        "C_p": res.C_p,
        "odbacene": odbacene,
    }


def main():
    meren = _merenja()
    ocene = [profil(b, meren) for b in BETE]
    lo, hi = UVERLJIVA_DUZINA_KM
    uverljive = [o for o in ocene if lo <= o["duzina"] <= hi]
    izbor = uverljive[-1] if uverljive else ocene[0]

    red = [
        "# Kalibracija tražnje za Novi Sad", "",
        "Cilj je bio podesiti `beta` u gravitacionom modelu na 18 izmerenih opterećenja",
        f"linija iz brojanja 2017 (ukupno {sum(meren.values()):,.0f} vožnji radnim danom)."
        .replace(",", "."),
        "**To ne uspeva, i razlog nije u tražnji.**", "",
        "| beta | prosečna dužina putovanja | udeo < 2 km | ulazaka po putovanju "
        "| linija bez putnika | Spearman vs brojanje |",
        "|---|---|---|---|---|---|",
    ]
    for o in ocene:
        red.append(f"| {o['beta']:.1f} | {o['duzina']:.2f} km | {100 * o['kratka']:.0f}% "
                   f"| {o['odnos']:.3f} | {o['praznih']} od {o['linija']} | {o['rho']:+.3f} |")

    red += [
        "", "## Zašto opterećenja po liniji ne mogu da kalibrišu beta", "",
        f"**Jedanaest od {ocene[0]['linija']} linija dobija tačno nula putnika, i to pri",
        "svakoj vrednosti bete.** Broj se ne menja jer uzrok nije raspodela tražnje nego",
        "dodela putnika: `assign` šalje ceo par najkraćim putem kroz mrežu linija, pa kad",
        "dve linije pokrivaju isti koridor, jedna uzme sve a druga ostane prazna. Stvarni",
        "putnici se raspoređuju na obe. Bez frekvencija i kapaciteta model nema čime da ih",
        "razdvoji, pa opterećenje po liniji nije identifikovano — ni jedna vrednost bete ga",
        "ne može popraviti.",
        "",
        "Posledica je i negativan Spearman: linija 9 je po brojanju najjača (19.879 vožnji),",
        "a u modelu dobija nulu jer joj varijanta 9A preuzme sve parove.",
        "",
        "Drugi, nezavisan problem: numeracija linija se između 2017. i današnjeg reda vožnje",
        "promenila. Brojanje daje liniji 18 svega 112 vožnji dnevno, a današnje 18A i 18B su",
        "među najdužim gradskim trasama. Poređenje po broju linije zato nije pouzdano ni",
        "kad bi dodela bila realistična.",
        "",
        "Da bi ova kalibracija imala smisla, treba (a) frekvencije i kapacitet u dodeli, što",
        "postoji u `core/frequencies.py` ali kao druga faza nad gotovom mrežom, i (b) mapiranje",
        "linija iz 2017 na današnje trase, koje traži red vožnje iz 2017 — a GSP ga ne čuva",
        "(vidi „Nema istorije\" u `docs/novi-sad.md`).",
        "",
        "## Šta je umesto toga urađeno", "",
        "**Beta se bira po dužini putovanja.** Pri `beta = 2.0`, vrednosti koja je do sada",
        f"stajala kao podrazumevana, prosečno putovanje je {ocene[4]['duzina']:.2f} km i",
        f"{100 * ocene[4]['kratka']:.0f}% tražnje pada na parove kraće od 2 km. To nije",
        "autobusko putovanje. Zone Novog Sada su guste, pa jako opadanje sa daljinom svu",
        "tražnju slepi za susedne zone.",
        "",
        f"Izabrano: **beta = {izbor['beta']:.1f}**, prosečna dužina {izbor['duzina']:.2f} km,",
        f"u opsegu {lo}-{hi} km koji je uverljiv za gradski autobus. Ovo je izbor po",
        "uverljivosti, ne kalibracija, i tako mora biti opisan u radu. Osetljivost na",
        "beta ide uz svaki rezultat za Novi Sad.",
        "",
        "**Ukupan obim je ispravljen.** Brojanje iz 2017 daje 172.687 *ulazaka* — putnik koji",
        "presedne broji se dvaput. Matrica nosi *putovanja*. Do sada je njen zbir bio",
        "postavljen na 172.687, što precenjuje tražnju za prosečan broj ulazaka po putovanju.",
    ]

    odnos = izbor["odnos"]
    ciljni = izvori.TREND_PUTNIKA[2017]
    putovanja = ciljni / odnos
    p_str = f"{putovanja:,.0f}".replace(",", ".")
    c_str = f"{ciljni:,.0f}".replace(",", ".")
    red += [
        "",
        f"Na GSP mreži model daje {odnos:.3f} ulazaka po putovanju (`d_0 = {izbor['d_0']:.3f}`),",
        f"pa je ispravan zbir matrice **{p_str} putovanja**, a ne {c_str}.",
        f"Time predviđeni ulasci pogađaju brojanje tačno: {p_str} x {odnos:.3f} = {c_str}.",
        "",
        "## Preostalo ograničenje", "",
        "Ovo je i dalje nekalibrisana matrica u smislu prostorne raspodele — poklapa se sa",
        "brojanjem po ukupnom obimu, ne i po tome ko kuda putuje. U radu ide kao ograničenje,",
        "zajedno sa gore opisanim razlogom zašto jača provera nije bila moguća.",
    ]
    if izbor["odbacene"]:
        red += ["",
                "Iz GSP mreže je izbačena linija "
                + ", ".join(f"`{o}` ({z} zona)" for o, _, z in izbor["odbacene"])
                + " jer se u zonskom grafu svodi na tačku."]

    out = Path(__file__).parent.parent.parent / "results" / "novisad_kalibracija.md"
    out.write_text("\n".join(red) + "\n", encoding="utf-8")
    print("\n".join(red))
    print(f"\nsnimljeno u {out}")
    print(f"\nUPISATI u traznja.py: BETA = {izbor['beta']}, PUTOVANJA = {putovanja:.0f}")


if __name__ == "__main__":
    main()
