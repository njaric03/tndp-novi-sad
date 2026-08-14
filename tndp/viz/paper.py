# Figure za seminarski rad, crtane iz vec snimljenih tabela u results/.
# Namerno se ne pokrecu eksperimenti ponovo: slika i tabela u radu moraju da
# pokazuju iste brojeve, a jedini nacin da to bude sigurno je jedan izvor.
# pokretanje: python -m tndp.viz.paper

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tndp.viz import style

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"


# markdown tabela iz results/*.md u listu recnika; prazna ako fajl nema tabelu
def _table(name):
    rows = []
    header = None
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if header is None:
            header = cells
            continue
        rows.append(dict(zip(header, cells)))
    return rows


# "1.630 ± 0.193" -> 1.630; "-" -> nan
def _number(s):
    s = s.split("±")[0].strip().replace("<", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# Kvalitet pod istim vremenskim budzetom. Bez ove slike kolona "s/grad" u
# glavnoj tabeli ne znaci nista: metode se tamo porede na razlicitom racunanju.
def budget(out):
    rows = _table("anytime.md")
    by_method = {}
    for r in rows:
        by_method.setdefault(r["metoda"], []).append(
            (_number(r["s/grad"]), _number(r["cilj"])))

    style.apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for name, points in by_method.items():
        points = np.array(sorted(points))
        color = style.color_for(name)
        if len(points) > 1:
            ax.plot(points[:, 0], points[:, 1], "o-", color=color,
                    label=style.label(name))
        else:
            ax.plot(points[:, 0], points[:, 1], "*", color=color, markersize=13,
                    label=style.label(name))
    ax.set_xscale("log")
    ax.set_xlabel("sekundi po gradu (logaritamska skala)")
    ax.set_ylabel("cilj, manja vrednost je bolja")
    ax.legend(loc="upper right")
    return style.save(fig, out)


# Levo klasican Pareto front, desno cilj po alfi. Front sam ne pokazuje gde se
# metode ukrstaju, a bas to je tvrdnja u tekstu, pa ide drugi panel.
def pareto(out):
    rows = _table("pareto.md")
    curves = {}
    for r in rows:
        curves.setdefault(r["metoda"], []).append(
            (_number(r["alpha"]), _number(r["C_p_all (min)"]),
             _number(r["C_o (min)"]), _number(r["cilj"])))

    style.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for name, t in curves.items():
        t = np.array(sorted(t))
        color = style.color_for(name)
        ax1.plot(t[:, 1], t[:, 2], "o-", color=color, label=style.label(name))
        ax2.plot(t[:, 0], t[:, 3], "o-", color=color, label=style.label(name))
    # alfa se ispisuje samo na jednoj krivoj i samo na svakoj drugoj tacki:
    # gornji kraj fronta je zbijen, pa bi se pune oznake preklopile
    reference = np.array(sorted(curves["hill climbing"]))
    for a, cp, co, _ in reference:
        if a not in (0.1, 0.25, 0.5, 0.9):
            continue
        ax1.annotate(f"$\\alpha$ = {a:g}", (cp, co), fontsize=9.0,
                     xytext=(6, 4), textcoords="offset points", color="#555555")
    # crta se C_p_all, ne C_p: nepokriveni parovi su naplaceni, inace bi metoda
    # koja ispusti najteze parove izgledala najbolje bas na ovoj osi
    ax1.set_xlabel("$C_{p,all}$, prosečno vreme putovanja (min)")
    ax1.set_ylabel("$C_o$, ukupno vreme linija (min)")
    ax1.set_title("front putnik/prevoznik, oznake su $\\alpha$", fontsize=11.5)
    ax1.legend(loc="upper right")

    # tacke ukrstanja politike sa klasicnim metodama, linearno izmedju uzoraka
    rl = np.array(sorted(curves["RL sampling 32"]))
    for opponent in ("greedy", "hill climbing"):
        p = np.array(sorted(curves[opponent]))
        diff = rl[:, 3] - p[:, 3]
        sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
        for k in sign_change:
            t = diff[k] / (diff[k] - diff[k + 1])
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
    print("->", *budget(RESULTS / "slika-budzet"))
    print("->", *pareto(RESULTS / "slika-pareto"))


if __name__ == "__main__":
    main()
