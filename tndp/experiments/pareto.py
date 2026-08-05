# Pareto front putnik/operater: sweep po alpha

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.experiments.common import (evaluate_method, held_out_cities,
                                     load_policy, scales_for)
from tndp.rl.evaluate import decode_sampling
from tndp.viz import style

ALPHAS = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cities", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    cities = held_out_cities(cfg, args.cities)
    scales = scales_for(cities)

    curves = {"greedy": [], "hill climbing": [], "RL sampling 32": []}
    rows = ["| alpha | metoda | cilj | C_p_all (min) | C_o (min) | d_un |",
            "|---|---|---|---|---|---|"]
    for a in args.alphas:
        methods = {
            "greedy": lambda c: greedy_network(c, R, lo, hi, alpha=a)[0],
            "hill climbing": lambda c: hill_climb(c, R, lo, hi, alpha=a)[0],
            "RL sampling 32": lambda c: decode_sampling(policy, c, R, k=32,
                                                        min_len=lo, max_len=hi,
                                                        alpha=a)[0],
        }
        for name, solve in methods.items():
            s = evaluate_method(solve, cities, scales, R, lo, hi, a)
            curves[name].append((s["C_p_all"].mean(), s["C_o"].mean()))
            rows.append(f"| {a} | {name} | {s['cilj'].mean():.3f} "
                        f"| {s['C_p_all'].mean():.2f} | {s['C_o'].mean():.0f} "
                        f"| {s['d_un'].mean():.3f} |")
            print(rows[-1])

    style.primeni()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for (name, pts), color, marker in zip(curves.items(),
                                          ["tab:orange", "tab:red", "tab:blue"],
                                          ["s", "^", "o"]):
        pts = np.array(pts)
        order = np.argsort(pts[:, 0])
        ax.plot(pts[order, 0], pts[order, 1], marker + "-", color=color, label=name)
        for (cp, co), a in zip(pts, args.alphas):
            ax.annotate(f"{a}", (cp, co), fontsize=7, xytext=(3, 3),
                        textcoords="offset points", color=color)
    ax.set_xlabel("C_p_all, prosečno vreme putovanja (min), manje bolje")
    ax.set_ylabel("C_o, ukupno vreme linija (min), manje bolje")
    ax.set_title(f"Pareto front putnik/operater, sweep po alpha\n"
                 f"({args.cities} held-out gradova, R={R}); "
                 f"RL je JEDNA politika uslovljena na alpha")
    ax.legend()
    ax.grid(alpha=0.3)

    results = Path(__file__).parent.parent.parent / "results"
    fig.tight_layout()
    fig.savefig(results / "pareto.png", dpi=130)
    header = [f"# Pareto front ({args.cities} held-out gradova, R={R}, "
              f"model {args.checkpoint})", "",
              "Ista trenirana politika je puštena na svaku vrednost alpha "
              "(alpha je feature čvora i uzorkuje se tokom treninga);",
              "baselines se za svaku tačku pokreću iznova.", ""]
    (results / "pareto.md").write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    print(f"snimljeno u {results / 'pareto.md'} i pareto.png")


if __name__ == "__main__":
    main()
