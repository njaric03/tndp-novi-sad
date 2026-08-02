# Kvalitet u funkciji utrošenog vremena. Bez ovoga poređenje "random
# najbolja od 200" protiv "RL sampling 32" ne znači ništa — budžeti su
# različiti i nigde se ne izveštavaju. Ovo je jedina fer forma poređenja
# metoda koje se skaliraju sa vremenom.
#
# pokretanje: python -m tndp.experiments.anytime runs/gravity-v1/best.pt

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign, objective
from tndp.experiments.common import held_out_cities, load_policy, scales_for
from tndp.rl.evaluate import decode, decode_sampling
from tndp.rl.mcts import mcts_decode

SAMPLES_K = [1, 2, 4, 8, 16, 32, 64]
RANDOM_N = [25, 100, 400, 1600, 6400]
CLIMB_EVALS = [100, 400, 1600, 6400]
MCTS_SIMS = [10, 25, 50, 100]


def run(solve, cities, scales, R, lo, hi, a):
    objs, t0 = [], time.perf_counter()
    for c, sc in zip(cities, scales):
        net = solve(c)
        assert net.check(c, R, lo, hi) == [], net.check(c, R, lo, hi)
        objs.append(objective(assign(c, net, compute_transfers=False), sc, a))
    return (time.perf_counter() - t0) / len(cities), float(np.mean(objs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cities", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--mcts", action="store_true", help="uključi MCTS (sporo)")
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    a = args.alpha if args.alpha is not None else cfg["alpha_eval"]
    cities = held_out_cities(cfg, args.cities)
    scales = scales_for(cities)

    series = {
        "random search": [(n, lambda c, n=n: random_search(
            c, R, lo, hi, num_samples=n, alpha=a)[0]) for n in RANDOM_N],
        "hill climbing": [(n, lambda c, n=n: hill_climb(
            c, R, lo, hi, alpha=a, max_evals=n)[0]) for n in CLIMB_EVALS],
        "RL sampling": [(k, lambda c, k=k: decode_sampling(
            policy, c, R, k=k, min_len=lo, max_len=hi, alpha=a)[0]) for k in SAMPLES_K],
    }
    if args.mcts:
        series["RL + MCTS"] = [(s, lambda c, s=s: mcts_decode(
            policy, c, R, lo, hi, a, sims=s)[0]) for s in MCTS_SIMS]

    curves, rows = {}, ["| metoda | budžet | s/grad | cilj |", "|---|---|---|---|"]
    for name, budgets in series.items():
        pts = []
        for budget, solve in budgets:
            dt, obj = run(solve, cities, scales, R, lo, hi, a)
            pts.append((dt, obj))
            rows.append(f"| {name} | {budget} | {dt:.3f} | {obj:.3f} |")
            print(rows[-1])
        curves[name] = np.array(pts)

    # jednokratne metode kao tačke
    for name, solve in [("greedy", lambda c: greedy_network(c, R, lo, hi, alpha=a)[0]),
                        ("RL greedy dekod", lambda c: decode(policy, c, R, lo, hi, a)[0])]:
        dt, obj = run(solve, cities, scales, R, lo, hi, a)
        curves[name] = np.array([[dt, obj]])
        rows.append(f"| {name} | 1 | {dt:.3f} | {obj:.3f} |")
        print(rows[-1])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, pts in curves.items():
        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], "o-", label=name)
        else:
            ax.plot(pts[:, 0], pts[:, 1], "*", ms=14, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("sekundi po gradu (log)")
    ax.set_ylabel("cilj (manje je bolje)")
    ax.set_title(f"Kvalitet pod istim vremenskim budžetom "
                 f"({args.cities} held-out gradova, alpha={a})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    results = Path(__file__).parent.parent.parent / "results"
    fig.tight_layout()
    fig.savefig(results / "anytime.png", dpi=130)
    (results / "anytime.md").write_text(
        "\n".join([f"# Anytime poređenje ({args.cities} gradova, alpha={a}, "
                   f"model {args.checkpoint})", ""] + rows) + "\n", encoding="utf-8")
    print(f"snimljeno u {results / 'anytime.md'} i anytime.png")


if __name__ == "__main__":
    main()
