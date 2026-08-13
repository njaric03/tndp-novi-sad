# Osetljivost zakljucaka na dve konstante koje u funkciji cilja ostaju stvar izbora: UNSERVED_FACTOR i alpha

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tndp.core.assignment as A
from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign, cost_scales, objective
from tndp.viz.style import color_for, save
from tndp.viz import style

FACTORS = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
ALPHAS = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]


def methods_for(policy, cfg, a):
    from tndp.rl.evaluate import decode_sampling
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    return {
        "random 200": lambda c: random_search(c, R, lo, hi, num_samples=200,
                                              alpha=a)[0],
        "greedy": lambda c: greedy_network(c, R, lo, hi, alpha=a)[0],
        "hill climbing": lambda c: hill_climb(c, R, lo, hi, alpha=a)[0],
        "RL sampling 32": lambda c: decode_sampling(policy, c, R, k=32,
                                                    min_len=lo, max_len=hi,
                                                    alpha=a)[0],
    }


def mean_objective(solve, cities, a):
    out = []
    for c in cities:
        net = solve(c)
        out.append(objective(assign(c, net, compute_transfers=False),
                             cost_scales(c), a))
    return float(np.mean(out))


def sweep(policy, cfg, cities, values, kind):
    curves = {}
    base = A.UNSERVED_FACTOR
    try:
        for v in values:
            a = cfg["alpha_eval"] if kind == "factor" else v
            if kind == "factor":
                A.UNSERVED_FACTOR = v
            for name, solve in methods_for(policy, cfg, a).items():
                curves.setdefault(name, []).append(
                    mean_objective(solve, cities, a))
            print(f"  {kind}={v}: " + ", ".join(
                f"{n} {curves[n][-1]:.3f}" for n in curves))
    finally:
        A.UNSERVED_FACTOR = base
    return curves


def panel(ax, values, curves, xlabel, marker_at=None):
    for name, ys in curves.items():
        ax.plot(values, ys, "o-", ms=4, lw=1.8, label=name,
                color=color_for(name))
    # tacke gde se dve metode presecaju: tu ranziranje zavisi od konstante
    names = list(curves)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.array(curves[names[i]]) - np.array(curves[names[j]])
            for k in range(len(d) - 1):
                if d[k] == 0 or d[k] * d[k + 1] < 0:
                    t = abs(d[k]) / (abs(d[k]) + abs(d[k + 1]))
                    x = values[k] + t * (values[k + 1] - values[k])
                    y = curves[names[i]][k] + t * (
                        curves[names[i]][k + 1] - curves[names[i]][k])
                    ax.plot([x], [y], "x", color="black", ms=9, mew=2,
                            zorder=5)
                    ax.annotate(f"{x:.2f}", (x, y), fontsize=7,
                                textcoords="offset points", xytext=(4, -10))
    if marker_at is not None:
        ax.axvline(marker_at, color="0.5", ls=":", lw=1.2)
        ax.annotate("u upotrebi", (marker_at, ax.get_ylim()[1]), fontsize=7,
                    color="0.4", rotation=90, va="top",
                    textcoords="offset points", xytext=(3, -4))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cilj (manje je bolje)")
    ax.grid(alpha=0.3)


def main():
    from tndp.experiments.common import held_out_cities, load_policy
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cities", type=int, default=8)
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    cities = held_out_cities(cfg, args.cities)

    print("sweep po UNSERVED_FACTOR:")
    by_factor = sweep(policy, cfg, cities, FACTORS, "factor")
    print("sweep po alpha:")
    by_alpha = sweep(policy, cfg, cities, ALPHAS, "alpha")

    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panel(axes[0], FACTORS, by_factor, "UNSERVED_FACTOR (naplata nepokrivenog para)",
          marker_at=A.UNSERVED_FACTOR)
    axes[0].set_title("osetljivost na naplatu nepokrivene tražnje")
    panel(axes[1], ALPHAS, by_alpha, "alpha (težina putničkog člana)",
          marker_at=cfg["alpha_eval"])
    axes[1].set_title("osetljivost na kompromis putnik/operater")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Zavisi li rangiranje metoda od izabranih konstanti "
                 f"({args.cities} gradova)", fontsize=11)

    out = Path(__file__).parent.parent.parent / "results"
    out.mkdir(exist_ok=True)
    print("snimljeno u " + ", ".join(save(fig, out / "sensitivity")))


if __name__ == "__main__":
    main()
