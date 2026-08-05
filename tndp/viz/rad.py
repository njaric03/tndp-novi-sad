# Figure za seminarski rad, crtane iz vec snimljenih tabela u results/.
# Namerno se ne pokrecu eksperimenti ponovo: slika i tabela u radu moraju da
# pokazuju iste brojeve, a jedini nacin da to bude sigurno je jedan izvor.
# pokretanje: python -m tndp.viz.rad

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tndp.viz import style

KOREN = Path(__file__).resolve().parent.parent.parent
REZULTATI = KOREN / "results"


# markdown tabela iz results/*.md u listu recnika; prazna ako fajl nema tabelu
def _tabela(ime):
    redovi = []
    zaglavlje = None
    for red in (REZULTATI / ime).read_text(encoding="utf-8").splitlines():
        red = red.strip()
        if not red.startswith("|"):
            continue
        celije = [c.strip() for c in red.strip("|").split("|")]
        if set("".join(celije)) <= set("-: "):
            continue
        if zaglavlje is None:
            zaglavlje = celije
            continue
        redovi.append(dict(zip(zaglavlje, celije)))
    return redovi


# "1.630 ± 0.193" -> 1.630; "-" -> nan
def _broj(s):
    s = s.split("±")[0].strip().replace("<", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# Kvalitet pod istim vremenskim budzetom. Bez ove slike kolona "s/grad" u
# glavnoj tabeli ne znaci nista: metode se tamo porede na razlicitom racunanju.
def budzet(out):
    red = _tabela("anytime.md")
    po_metodi = {}
    for r in red:
        po_metodi.setdefault(r["metoda"], []).append(
            (_broj(r["s/grad"]), _broj(r["cilj"])))

    style.primeni()
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for ime, tacke in po_metodi.items():
        tacke = np.array(sorted(tacke))
        boja = style.color_for(ime)
        if len(tacke) > 1:
            ax.plot(tacke[:, 0], tacke[:, 1], "o-", color=boja,
                    label=style.naziv(ime))
        else:
            ax.plot(tacke[:, 0], tacke[:, 1], "*", color=boja, markersize=13,
                    label=style.naziv(ime))
    ax.set_xscale("log")
    ax.set_xlabel("sekundi po gradu (logaritamska skala)")
    ax.set_ylabel("cilj, manja vrednost je bolja")
    ax.legend(loc="upper right")
    return style.save(fig, out)


# Levo klasican Pareto front, desno cilj po alfi. Front sam ne pokazuje gde se
# metode ukrstaju, a bas to je tvrdnja u tekstu, pa ide drugi panel.
def pareto(out):
    red = _tabela("pareto.md")
    krive = {}
    for r in red:
        krive.setdefault(r["metoda"], []).append(
            (_broj(r["alpha"]), _broj(r["C_p_all (min)"]),
             _broj(r["C_o (min)"]), _broj(r["cilj"])))

    style.primeni()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ime, t in krive.items():
        t = np.array(sorted(t))
        boja = style.color_for(ime)
        ax1.plot(t[:, 1], t[:, 2], "o-", color=boja, label=style.naziv(ime))
        ax2.plot(t[:, 0], t[:, 3], "o-", color=boja, label=style.naziv(ime))
    # alfa se ispisuje samo na jednoj krivoj i samo na svakoj drugoj tacki:
    # gornji kraj fronta je zbijen, pa bi se pune oznake preklopile
    referentna = np.array(sorted(krive["hill climbing"]))
    for a, cp, co, _ in referentna:
        if a not in (0.1, 0.25, 0.5, 0.9):
            continue
        ax1.annotate(f"$\\alpha$ = {a:g}", (cp, co), fontsize=9.0,
                     xytext=(6, 4), textcoords="offset points", color="#555555")
    ax1.set_xlabel("$C_p$, prosečno vreme putovanja (min)")
    ax1.set_ylabel("$C_o$, ukupno vreme linija (min)")
    ax1.set_title("front putnik/prevoznik, oznake su $\\alpha$", fontsize=11.5)
    ax1.legend(loc="upper right")

    # tacke ukrstanja politike sa klasicnim metodama, linearno izmedju uzoraka
    rl = np.array(sorted(krive["RL sampling 32"]))
    for protivnik in ("greedy", "hill climbing"):
        p = np.array(sorted(krive[protivnik]))
        razlika = rl[:, 3] - p[:, 3]
        znak = np.where(np.diff(np.sign(razlika)) != 0)[0]
        for k in znak:
            t = razlika[k] / (razlika[k] - razlika[k + 1])
            a = rl[k, 0] + t * (rl[k + 1, 0] - rl[k, 0])
            j = rl[k, 3] + t * (rl[k + 1, 3] - rl[k, 3])
            ax2.plot([a], [j], "kx", markersize=9, markeredgewidth=1.8, zorder=5)
            ax2.annotate(f"$\\alpha$ = {a:.2f}", (a, j), fontsize=9.5,
                         xytext=(6, -13), textcoords="offset points")
    ax2.set_xlabel("$\\alpha$, težina putničkog člana")
    ax2.set_ylabel("cilj, manja vrednost je bolja")
    ax2.set_title("isti podaci kao levo, ali po $\\alpha$", fontsize=11.5)
    return style.save(fig, out)


def main():
    print("->", *budzet(REZULTATI / "slika-budzet"))
    print("->", *pareto(REZULTATI / "slika-pareto"))


if __name__ == "__main__":
    main()
