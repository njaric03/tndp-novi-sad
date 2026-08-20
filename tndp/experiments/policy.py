# Sta je politika naucila, a ne samo sta je proizvela

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from tndp.core.network import TransitNetwork
from tndp.rl.env import HALT, TndpEnv
from tndp.rl.features import edge_tensors, node_features
from tndp.viz.maps import draw_network
from tndp.viz.style import save
from tndp.viz import style
from tndp import RESULTS


# 1 cvor, 2-4 cvora, 5+ cvorova - naslov panela se inace procita kao greska
def _cvorova(k):
    if k % 10 == 1 and k % 100 != 11:
        return "čvor"
    if k % 10 in (2, 3, 4) and k % 100 not in (12, 13, 14):
        return "čvora"
    return "čvorova"


# ulice u pozadini, isto kao u maps.draw_network ali bez linija
def _streets(ax, city):
    for i, j in city.street_edges:
        ax.plot(city.coords[[i, j], 0], city.coords[[i, j], 1],
                color="0.88", lw=0.8, zorder=1)


# jedan prolaz politike kroz tekuce stanje; decision() se zove tacno jednom, pa
# maska po kojoj se racunaju verovatnoce i ona po kojoj se bira potez ne mogu da se raziđu
@torch.no_grad()
def _forward(policy, env, edge_index, edge_attr):
    decision, mask = env.decision()
    h = policy.encode(node_features(env, policy.features), edge_index, edge_attr)
    return decision, policy.action_logits(h, decision, mask, env.ends), mask


# verovatnoce sledeceg poteza po cvoru
def step_probs(policy, env, edge_index, edge_attr):
    decision, logits, _ = _forward(policy, env, edge_index, edge_attr)
    p = torch.softmax(logits, dim=0).numpy()
    n = env.city.n
    halt = float(p[-1]) if decision == HALT else 0.0
    return p[:2 * n].reshape(2, n).sum(0), halt, decision


def heatmap(policy, city, cfg, alpha, out):
    env = TndpEnv(city, cfg["num_routes"], cfg["min_len"], cfg["max_len"], alpha)
    edge_index, edge_attr = edge_tensors(city)
    env.reset()
    snaps = []
    while not env.done:
        decision, logits, mask = _forward(policy, env, edge_index, edge_attr)
        if len(env.routes) == 0:
            p = torch.softmax(logits, dim=0).numpy()
            n = city.n
            halt = float(p[-1]) if decision == HALT else 0.0
            # cvor je dozvoljen ako se sme dodati sa bar jednog kraja linije;
            # bez toga se maskiran i dozvoljen-ali-neverovatan cvor crtaju isto
            dozvoljen = np.asarray(mask).reshape(2, n).any(0)
            snaps.append((env.current[:], p[:2 * n].reshape(2, n).sum(0), halt,
                          dozvoljen))
        a = int(logits.argmax())
        env.step(-1 if (decision == HALT and a == len(logits) - 1) else a)

    # stanja gde je politika vec odlucila da stane se izbacuju - nemaju sta da
    # pokazu na mapi po cvorovima
    usable = [i for i, s in enumerate(snaps) if s[2] < 0.99]
    # Prvi panel je uvek prvi potez: tu su svi cvorovi dozvoljeni, pa raspodela
    # govori o politici a ne o maski. Drugi je potez sa najvise dozvoljenih
    # cvorova, jer pri kraju linije maska cesto ostavi jedan jedini potez -
    # verovatnoca 1 tamo ne bi bila nalaz o politici nego o pravilima.
    kasniji = max(usable[1:], key=lambda i: snaps[i][3].sum())
    pick = [usable[0], kasniji]
    # jedna skala boje za sve panele da se mogu porediti
    vmax = max(snaps[si][1].max() for si in pick)
    # velicina cvora = traznja u njemu - bez toga se ne vidi da li politika
    # prati traznju ili samo geometriju
    mass = city.demand.sum(0) + city.demand.sum(1)
    size = 55 + 240 * (mass - mass.min()) / max(mass.max() - mass.min(), 1e-9)

    style.apply_style()
    fig, axes = plt.subplots(1, len(pick), figsize=(4.3 * len(pick), 4.3))
    axes = np.atleast_1d(axes)
    for ax, si in zip(axes, pick):
        cur, probs, halt, dozvoljen = snaps[si]
        _streets(ax, city)
        # Maskiran cvor nije isto sto i dozvoljen cvor male verovatnoce, a na
        # jednoj skali boje izgledaju isto (oba skoro bela). Maskirani zato idu
        # sivo i izvan skale: citalac odmah vidi koliko je izbora uopste bilo.
        ax.scatter(city.coords[~dozvoljen, 0], city.coords[~dozvoljen, 1],
                   c="#e0e0e0", s=size[~dozvoljen], zorder=3,
                   edgecolors="#bdbdbd", linewidths=0.5)
        # svetlo-ka-tamnom umesto viridisa: nula treba da bude skoro bela, jer
        # je vecina cvorova u svakom potezu nedostupna ili neverovatna
        # redom po verovatnoci, da najverovatniji cvor ostane na vrhu: dva
        # najverovatnija na prvom potezu su susedi i krugovi im se preklapaju,
        # pa je crvenog dosad prekrivao narandzasti, koji je samo veci jer nosi
        # vise traznje. Obod je tamniji kod oznacenih, da se razdvoje i tako
        red = np.flatnonzero(dozvoljen)[np.argsort(probs[dozvoljen])]
        sc = ax.scatter(city.coords[red, 0], city.coords[red, 1],
                        c=probs[red],
                        cmap="YlOrRd", s=size[red], zorder=3, vmin=0.0,
                        vmax=vmax, linewidths=np.where(probs[red] >= 0.02, 1.2, 0.5),
                        edgecolors=np.where(probs[red] >= 0.02, "#1a1a1a", "#999999"))
        # izgradjen deo linije je plav, ne crven: crvena je gornji kraj skale
        # verovatnoce, pa bi se trasa stapala sa najverovatnijim cvorom
        if len(cur) > 1:
            pts = city.coords[cur]
            ax.plot(pts[:, 0], pts[:, 1], color="#1f4e9c", lw=2.6,
                    zorder=2, solid_capstyle="round")
        if cur:
            ax.scatter(city.coords[cur, 0], city.coords[cur, 1],
                       facecolors="none", edgecolors="#1f4e9c",
                       s=size[cur] + 90, lw=1.8, zorder=4)
        # Skala boje je zajednicka za oba panela, pa panel u kom je najveca
        # verovatnoca 0,6 izgleda prazniji nego sto jeste. Tri najverovatnija
        # cvora zato nose i broj: citalac vidi koliko je politika izostrena, a
        # ne mora da pogadja nijansu sa trake.
        # Tri najverovatnija cvora znaju da budu toliko blizu da im se krugovi
        # preklope (na prvom potezu dva najverovatnija su susedi), pa oznaka uz
        # sam cvor ne kaze kojem pripada: idu razmaknuto gore, dole i desno, uz
        # tanku vodilju do svog cvora.
        MESTA = [(0, 30, "center", "bottom"), (0, -30, "center", "top"),
                 (40, 0, "left", "center")]
        for mesto, k in zip(MESTA, np.argsort(probs)[::-1][:3]):
            if probs[k] < 0.02:
                continue
            dx, dy, ha, va = mesto
            ax.annotate(f"{probs[k]:.2f}".replace(".", ","),
                        city.coords[k], fontsize=9.5, zorder=6, ha=ha, va=va,
                        xytext=(dx, dy), textcoords="offset points",
                        color="#1a1a1a",
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color="#8a8a8a", shrinkA=2, shrinkB=3),
                        path_effects=[pe.withStroke(linewidth=2.4,
                                                    foreground="white")])
        # bez margine oznaka nad rubnim cvorom izadje van ose i bude odsecena
        ax.margins(0.09)
        stanje = ("linija još prazna" if not cur
                  else f"linija ima {len(cur)} {_cvorova(len(cur))}")
        ax.set_title(f"potez {si + 1}: {stanje}", fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
    cb = fig.colorbar(sc, ax=list(axes), fraction=0.02, pad=0.01,
                      label="verovatnoća da politika izabere taj čvor")
    cb.ax.yaxis.set_major_formatter(style.zarez_formatter())
    # Bez legende na samoj slici: cetiri reda teksta preko grafa pojedu mrezu, a
    # isto obavestenje staje u potpis figure u radu.
    return save(fig, out)


# Slika pokazuje jedan grad; ovo meri isto na svih 20 i daje broj koji se sme
# napisati u radu. Poredi se sa stepenom cvora da se vidi da nije rec o tome da
# politika samo bira dobro povezane cvorove.
def first_move_correlation(policy, cfg, alpha, cities, out):

    r_demand, r_degree = [], []
    for city in cities:
        env = TndpEnv(city, cfg["num_routes"], cfg["min_len"], cfg["max_len"], alpha)
        edge_index, edge_attr = edge_tensors(city)
        env.reset()
        probs, _, _ = step_probs(policy, env, edge_index, edge_attr)
        mass = city.demand.sum(0) + city.demand.sum(1)
        degree = np.isfinite(city.street_time).sum(1)
        r_demand.append(spearmanr(probs, mass).statistic)
        r_degree.append(spearmanr(probs, degree).statistic)

    rows = [f"# Prvi potez politike naspram tražnje ({len(cities)} gradova, "
            f"alpha={alpha})", "",
            "Spearmanova korelacija verovatnoće da politika izabere čvor kao",
            "početni, sa dve osobine tog čvora. Rangovi, ne vrednosti.", "",
            "| osobina čvora | prosečan rho | sd | pozitivan na |",
            "|---|---|---|---|"]
    for name, r in (("tražnja u čvoru", r_demand), ("stepen u uličnoj mreži", r_degree)):
        rows.append(f"| {name} | {np.mean(r):+.3f} | {np.std(r):.3f} | "
                    f"{sum(x > 0 for x in r)}/{len(r)} |")
    Path(out).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows[5:]))
    return [str(out)]


def filmstrip(policy, city, cfg, alpha, out):
    env = TndpEnv(city, cfg["num_routes"], cfg["min_len"], cfg["max_len"], alpha)
    edge_index, edge_attr = edge_tensors(city)
    env.reset()
    stages = []
    while not env.done:
        decision, logits, _ = _forward(policy, env, edge_index, edge_attr)
        a = int(logits.argmax())
        is_halt = decision == HALT and a == len(logits) - 1
        env.step(-1 if is_halt else a)
        if is_halt:
            stages.append([r[:] for r in env.routes])

    fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 4.4))
    axes = np.atleast_1d(axes)
    for ax, routes in zip(axes, stages):
        draw_network(ax, city, TransitNetwork(routes=routes),
                     title=f"posle {len(routes)}. linije")
    fig.suptitle(f"Epizoda gradi mrežu liniju po liniju, {city.name}, "
                 f"alpha={alpha}", fontsize=11)
    return save(fig, out)


def main():
    from tndp.experiments.common import held_out_cities, load_policy
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--city", type=int, default=0, help="indeks held-out grada")
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    a = args.alpha if args.alpha is not None else cfg["alpha_eval"]
    cities = held_out_cities(cfg, max(20, args.city + 1))
    city = cities[args.city]

    RESULTS.mkdir(exist_ok=True)
    print("snimljeno u " + ", ".join(
        heatmap(policy, city, cfg, a, RESULTS / "policy-heatmap")
        + filmstrip(policy, city, cfg, a, RESULTS / "filmstrip")
        + first_move_correlation(policy, cfg, a, cities,
                                 RESULTS / "policy-traznja.md")))


if __name__ == "__main__":
    main()
