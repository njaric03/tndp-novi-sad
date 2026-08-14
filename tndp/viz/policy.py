# Sta je politika naucila, a ne samo sta je proizvela

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tndp.core.network import TransitNetwork
from tndp.rl.env import HALT, TndpEnv
from tndp.rl.features import edge_tensors, node_features
from tndp.viz.maps import draw_network
from tndp.viz.style import save
from tndp.viz import style
from tndp import RESULTS


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
    return decision, policy.action_logits(h, decision, mask, env.ends)


# verovatnoce sledeceg poteza po cvoru
def step_probs(policy, env, edge_index, edge_attr):
    decision, logits = _forward(policy, env, edge_index, edge_attr)
    p = torch.softmax(logits, dim=0).numpy()
    n = env.city.n
    halt = float(p[-1]) if decision == HALT else 0.0
    return p[:2 * n].reshape(2, n).sum(0), halt, decision


def heatmap(policy, city, cfg, alpha, out, panels=2):
    env = TndpEnv(city, cfg["num_routes"], cfg["min_len"], cfg["max_len"], alpha)
    edge_index, edge_attr = edge_tensors(city)
    env.reset()
    snaps = []
    while not env.done:
        decision, logits = _forward(policy, env, edge_index, edge_attr)
        if len(env.routes) == 0:
            p = torch.softmax(logits, dim=0).numpy()
            n = city.n
            halt = float(p[-1]) if decision == HALT else 0.0
            snaps.append((env.current[:], p[:2 * n].reshape(2, n).sum(0), halt))
        a = int(logits.argmax())
        env.step(-1 if (decision == HALT and a == len(logits) - 1) else a)

    # stanja u kojima je politika vec odlucila da zavrsi liniju se izbacuju:
    # tamo je sva verovatnoca na potezu "kraj", pa mapa po cvorovima nema sta
    # da pokaze i panel samo trosi prostor
    usable = [i for i, s in enumerate(snaps) if s[2] < 0.99]
    pick = np.unique(np.linspace(0, len(usable) - 1, panels).astype(int))
    pick = [usable[k] for k in pick]
    # jedna skala boje za sve panele, inace se paneli ne mogu porediti
    vmax = max(snaps[si][1].max() for si in pick)
    # velicina cvora je ukupna traznja koja u njemu nastaje ili se zavrsava.
    # Bez nje se sa slike ne vidi ono sto je jedino zanimljivo: da li politika
    # bira cvorove sa traznjom ili samo geometrijski zgodne cvorove.
    mass = city.demand.sum(0) + city.demand.sum(1)
    size = 55 + 240 * (mass - mass.min()) / max(mass.max() - mass.min(), 1e-9)

    style.apply_style()
    fig, axes = plt.subplots(1, len(pick), figsize=(4.3 * len(pick), 4.3))
    axes = np.atleast_1d(axes)
    for ax, si in zip(axes, pick):
        cur, probs, halt = snaps[si]
        _streets(ax, city)
        # svetlo-ka-tamnom umesto viridisa: nula treba da bude skoro bela, jer
        # je vecina cvorova u svakom potezu nedostupna ili neverovatna
        sc = ax.scatter(city.coords[:, 0], city.coords[:, 1], c=probs,
                        cmap="YlOrRd", s=size, zorder=3, vmin=0.0, vmax=vmax,
                        edgecolors="#999999", linewidths=0.5)
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
        caption = ("linija još prazna" if not cur
                   else f"izgrađeno {len(cur)} čvorova")
        ax.set_title(f"potez {si + 1}: {caption}", fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(sc, ax=list(axes), fraction=0.02, pad=0.01,
                 label="verovatnoća da politika izabere taj čvor")
    fig.text(0.5, 0.035, "veličina čvora je tražnja koja u njemu nastaje ili se "
             "završava; plavo je već izgrađen deo linije;\nnedozvoljeni potezi "
             "su maskirani i imaju verovatnoću nula",
             ha="center", fontsize=10, color="#555555")
    return save(fig, out)


# Slika pokazuje jedan grad; ovo meri isto na svih 20 i daje broj koji se sme
# napisati u radu. Poredi se sa stepenom cvora da se vidi da nije rec o tome da
# politika samo bira dobro povezane cvorove.
def first_move_correlation(policy, cfg, alpha, cities, out):
    from scipy.stats import spearmanr

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
        decision, logits = _forward(policy, env, edge_index, edge_attr)
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
