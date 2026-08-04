# Šta je politika naučila, a ne samo šta je proizvela.
#
# Dve slike koje se prave iz istog rollout-a:
#   heatmap  — za nekoliko parcijalnih stanja tokom gradnje jedne linije,
#              čvorovi obojeni verovatnoćom da ih politika izabere sledeće,
#   filmstrip — mreža kako raste, jedna linija po panelu.
#
# Za predmet "Eksperimenti sa neuronskim mrežama" heatmapa je jedina slika
# koja pokazuje unutrašnjost modela; sve ostale prikazuju izlaz.
#
# pokretanje: python -m tndp.viz.policy runs/gravity-v1/best.pt

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
from tndp.viz.maps import LINE_COLORS, draw_network
from tndp.viz.style import save


# ulice u pozadini, isto kao u maps.draw_network ali bez linija
def _streets(ax, city):
    for i, j in city.street_edges:
        ax.plot(city.coords[[i, j], 0], city.coords[[i, j], 1],
                color="0.88", lw=0.8, zorder=1)


# verovatnoće sledećeg poteza po čvoru. akcija je par (kraj, čvor), pa se
# dve strane sabiraju: slika odgovara na "koliko je verovatno da linija
# uopšte ide u ovaj čvor", bez obzira na koji kraj se kači
@torch.no_grad()
def step_probs(policy, env, edge_index, edge_attr):
    decision, mask = env.decision()
    h = policy.encode(node_features(env, policy.version), edge_index, edge_attr)
    logits = policy.action_logits(h, decision, mask, env.ends)
    p = torch.softmax(logits, dim=0).numpy()
    n = env.city.n
    halt = float(p[-1]) if decision == HALT else 0.0
    return p[:2 * n].reshape(2, n).sum(0), halt, decision


def heatmap(policy, city, cfg, alpha, out, panels=4):
    env = TndpEnv(city, cfg["num_routes"], cfg["min_len"], cfg["max_len"], alpha)
    edge_index, edge_attr = edge_tensors(city)
    env.reset()
    snaps = []
    while not env.done:
        probs, halt, decision = step_probs(policy, env, edge_index, edge_attr)
        if len(env.routes) == 0:
            snaps.append((env.current[:], probs.copy(), halt))
        h = policy.encode(node_features(env, policy.version), edge_index, edge_attr)
        logits = policy.action_logits(h, decision, env.decision()[1], env.ends)
        a = int(logits.argmax())
        env.step(-1 if (decision == HALT and a == len(logits) - 1) else a)

    pick = np.unique(np.linspace(0, len(snaps) - 1, panels).astype(int))
    # jedna skala boje za sve panele: kad politika odluči da završi liniju,
    # verovatnoće po čvorovima padnu na ~1e-6 i zasebni kolorbar bi tu
    # sitnicu razvukao preko cele skale kao da je signal
    vmax = max(snaps[si][1].max() for si in pick)
    fig, axes = plt.subplots(1, len(pick), figsize=(3.8 * len(pick), 4.4))
    axes = np.atleast_1d(axes)
    for ax, si in zip(axes, pick):
        cur, probs, halt = snaps[si]
        _streets(ax, city)
        sc = ax.scatter(city.coords[:, 0], city.coords[:, 1], c=probs,
                        cmap="viridis", s=90, zorder=3, vmin=0.0, vmax=vmax)
        if len(cur) > 1:
            pts = city.coords[cur]
            ax.plot(pts[:, 0], pts[:, 1], color=LINE_COLORS[0], lw=2.6,
                    zorder=2, solid_capstyle="round")
        if cur:
            ax.scatter(city.coords[cur, 0], city.coords[cur, 1],
                       facecolors="none", edgecolors=LINE_COLORS[0], s=170,
                       lw=1.8, zorder=4)
        title = f"potez {si + 1}, linija {cur if cur else '—'}"
        if halt > 0:
            title += f"\nP(završi liniju) = {halt:.2f}"
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(sc, ax=list(axes), fraction=0.02, pad=0.01,
                 label="P(sledeći čvor)")
    fig.suptitle(f"Šta politika gleda dok gradi prvu liniju — {city.name}, "
                 f"alpha={alpha}", fontsize=11)
    return save(fig, out)


def filmstrip(policy, city, cfg, alpha, out):
    env = TndpEnv(city, cfg["num_routes"], cfg["min_len"], cfg["max_len"], alpha)
    edge_index, edge_attr = edge_tensors(city)
    env.reset()
    stages = []
    while not env.done:
        decision, mask = env.decision()
        h = policy.encode(node_features(env, policy.version), edge_index, edge_attr)
        logits = policy.action_logits(h, decision, mask, env.ends)
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
    fig.suptitle(f"Epizoda gradi mrežu liniju po liniju — {city.name}, "
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
    city = held_out_cities(cfg, args.city + 1)[args.city]

    out = Path(__file__).parent.parent.parent / "results"
    out.mkdir(exist_ok=True)
    print("snimljeno u " + ", ".join(
        heatmap(policy, city, cfg, a, out / "policy-heatmap")
        + filmstrip(policy, city, cfg, a, out / "filmstrip")))


if __name__ == "__main__":
    main()
