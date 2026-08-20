# Figure za seminarski rad, crtane iz vec snimljenih tabela u results/.
# Namerno se ne pokrecu eksperimenti ponovo: slika i tabela u radu moraju da
# pokazuju iste brojeve, a jedini nacin da to bude sigurno je jedan izvor.
# pokretanje: python -m tndp.viz.paper

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation

from tndp.viz import style
from tndp import RESULTS


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


# vrednosti po gradu iz anytime-po-gradu.md: {metoda: (vremena, matrica
# [budzet x grad])}. Bez toga se razlika dve metode moze prikazati samo kao
# razmak dva proseka, a to nije mera koju tabele u radu koriste.
def _po_gradu(name="anytime-po-gradu.md"):
    sirovo = {}
    for r in _table(name):
        sirovo.setdefault(r["metoda"], {}).setdefault(
            (_number(r["s/grad"]), int(r["budžet"])), {})[int(r["grad"])] = _number(r["cilj"])
    izlaz = {}
    for metoda, po_budzetu in sirovo.items():
        kljucevi = sorted(po_budzetu)
        gradovi = sorted(po_budzetu[kljucevi[0]])
        izlaz[metoda] = (np.array([t for t, _ in kljucevi]),
                         np.array([[po_budzetu[k][g] for g in gradovi]
                                   for k in kljucevi]))
    return izlaz


# vrednost metode u trenutku t, po gradu, linearnom interpolacijom po log-vremenu
# (budzeti idu geometrijski, pa je log prirodna osa). Van izmerenog raspona se ne
# ekstrapolira - pozivalac bira mrezu vremena unutar preseka svih metoda.
def _u_trenutku(vremena, vrednosti, t):
    return np.array([np.interp(np.log(t), np.log(vremena), vrednosti[:, g])
                     for g in range(vrednosti.shape[1])])


# Kvalitet pod istim vremenskim budzetom. Bez ove slike kolona "s/grad" u
# glavnoj tabeli ne znaci nista: metode se tamo porede na razlicitom racunanju.
#
# Crtaju se samo proseci, bez pojasa: gradovi se po tezini razlikuju vise nego
# metode medju sobom, pa se +-1 sd dve metode preklapa i kad je razlika na svakom
# pojedinacnom gradu ista. Razlika se zato meri uparano i ispisuje kao broj.
#
# Osa je prelomljena. Nasumicna pretraga stoji izmedju 1,9 i 2,5, a cela
# stvarna utakmica se igra izmedju 1,45 i 1,65; na jednoj osi ona zauzme gornju
# polovinu slike i sabije bas ono sto se poredi. Gore ide ona, dole svi ostali.
def budget(out):
    rows = _table("anytime.md")
    by_method = {}
    for r in rows:
        by_method.setdefault(r["metoda"], []).append(
            (_number(r["s/grad"]), _number(r["cilj"]), _number(r.get("sd", "-"))))

    style.apply_style()
    fig, (gore, dole) = plt.subplots(
        2, 1, sharex=True, figsize=(5.6, 4.1),
        gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.09})

    def nacrtaj(ax, name, points):
        color = style.color_for(name)
        if len(points) > 1:
            ax.plot(points[:, 0], points[:, 1], "o-", color=color,
                    label=style.label(name), markersize=4)
        else:
            ax.plot(points[:, 0], points[:, 1], "*", color=color, markersize=13,
                    label=style.label(name))

    # Gore idu dve slabe klasicne metode, dole one koje se stvarno takmice.
    # Podela je po metodi a ne po vrednosti, da prelom ostane na istom mestu i
    # kad se tabela pregenerise sa nesto drugacijim brojevima.
    GORE = ("random", "greedy")

    def na_vrhu(name):
        # "RL greedy dekod" pocinje na "rl", pa ne upada u "greedy"
        return name.lower().startswith(GORE)

    svi = {n: np.array(sorted(p)) for n, p in by_method.items()}
    for name, points in svi.items():
        nacrtaj(gore if na_vrhu(name) else dole, name, points)

    dole.set_xscale("log")

    # granice se racunaju iz podataka, da prelom ne mora rucno da se pomera kad
    # se tabela pregenerise
    def opseg(gornji):
        v = np.concatenate([t[:, 1] for n, t in svi.items()
                            if na_vrhu(n) == gornji])
        return float(np.nanmin(v)), float(np.nanmax(v))

    lo, hi = opseg(False)
    dole.set_ylim(lo - 0.04, hi + 0.04)
    lo, hi = opseg(True)
    gore.set_ylim(lo - 0.06, hi + 0.06)

    gore.spines["bottom"].set_visible(False)
    dole.spines["top"].set_visible(False)
    gore.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # kose crtice na mestu preloma, da se osa ne procita kao neprekidna
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7, linestyle="none",
              color="#666666", mec="#666666", mew=1.0, clip_on=False)
    # samo uz levu osu: desne ose nema, pa je crtica tamo visila u belini
    gore.plot([0], [0], transform=gore.transAxes, **kw)
    dole.plot([0], [1], transform=dole.transAxes, **kw)

    dole.set_xlabel("sekundi po gradu (logaritamska skala)")
    dole.set_ylabel("cilj, manja vrednost je bolja")
    dole.yaxis.set_label_coords(-0.1, 0.72)
    rucke = [*gore.get_legend_handles_labels()[0], *dole.get_legend_handles_labels()[0]]
    natpisi = [*gore.get_legend_handles_labels()[1], *dole.get_legend_handles_labels()[1]]
    # bez fontsize: rcParams je vec podesen tako da posle skaliranja na sirinu
    # stranice ispadne oko 8 pt, a rucno smanjenje to obara ispod citljivog
    dole.legend(rucke, natpisi, loc="lower left", ncol=1)

    # Uparena razlika se ne crta, jer bi referentna metoda bila ravna nula
    # kroz ceo panel, nego se ispisuje: brojevi iz ove poruke stoje u tekstu rada.
    po_gradu = _po_gradu()
    REFERENCA = "hill climbing"
    krive = {n: v for n, v in po_gradu.items() if len(v[0]) > 1}
    # mreza vremena je presek izmerenih raspona: van njega bi se ekstrapoliralo
    t_lo = max(v[0].min() for v in krive.values())
    t_hi = min(v[0].max() for v in krive.values())
    mreza = np.geomspace(t_lo, t_hi, 25)
    vremena_ref, vrednosti_ref = krive[REFERENCA]
    for name, (vremena, vrednosti) in krive.items():
        if name == REFERENCA:
            continue
        razlike = np.array([
            _u_trenutku(vremena, vrednosti, t)
            - _u_trenutku(vremena_ref, vrednosti_ref, t) for t in mreza])
        prosek = razlike.mean(axis=1)
        # standardna greska UPARENE razlike, ne sd po gradovima
        se = razlike.std(axis=1, ddof=1) / np.sqrt(razlike.shape[1])
        print(f"   uparena razlika {name} - {REFERENCA} na {t_lo:.2f}-{t_hi:.2f} s: "
              f"{prosek.min():+.3f} do {prosek.max():+.3f} (se {se.mean():.3f})")

    style.decimal_comma(gore, dole)
    # x-osa je logaritamska i nosi 10^-2, 10^0: te oznake ostaju u eksponentu
    dole.xaxis.set_major_formatter(LogFormatterSciNotation())
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
    # gornji kraj fronta je zbijen, pa bi se pune oznake preklopile.
    # Pomeraj je po tacki, ne jedinstven: front je konveksan, pa isti pomeraj
    # koji je kod alfa 0,5 pored krive kod 0,25 padne na nju, a kod 0,1 i 0,9
    # izadje van okvira ose.
    # Jedini pouzdano prazan deo panela je gore desno: tamo bi mreza bila i
    # spora i skupa, pa nijedna metoda ne stize dotle. Oznake zato idu u tom
    # smeru, a one koje se time udalje od svoje tacke dobijaju tanku vodilju.
    MESTO = {0.1: (-12, 32, "right", True), 0.25: (20, 16, "left", True),
             0.5: (11, 1, "left", False), 0.9: (12, -4, "left", False)}
    reference = np.array(sorted(curves["hill climbing"]))
    for a, cp, co, _ in reference:
        if a not in MESTO:
            continue
        dx, dy, ha, vodilja = MESTO[a]
        ax1.annotate(f"$\\alpha$ = {style.broj(a, 2)}", (cp, co), fontsize=9.0,
                     xytext=(dx, dy), textcoords="offset points", ha=ha,
                     va="center", color="#555555", zorder=6,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="#aaaaaa",
                                     shrinkA=1, shrinkB=4) if vodilja else None,
                     # bela kontura: kratke oznake svejedno prolaze uz krivu
                     path_effects=[pe.withStroke(linewidth=2.4,
                                                 foreground="white")])
    ax1.margins(y=0.06)
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
            ax2.annotate(f"$\\alpha$ = {style.broj(a, 2)}", (a, j), fontsize=9.5,
                         xytext=(6, -13), textcoords="offset points", zorder=6,
                         path_effects=[pe.withStroke(linewidth=2.4,
                                                     foreground="white")])
    ax2.set_xlabel("$\\alpha$, težina putničkog člana")
    ax2.set_ylabel("cilj, manja vrednost je bolja")
    ax2.set_title("isti podaci kao levo, ali po $\\alpha$", fontsize=11.5)
    style.decimal_comma(ax1, ax2)
    return style.save(fig, out)


def main():
    print("->", *budget(RESULTS / "slika-budzet"))
    print("->", *pareto(RESULTS / "slika-pareto"))


if __name__ == "__main__":
    main()
