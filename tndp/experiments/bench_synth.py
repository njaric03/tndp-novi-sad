# RL politika protiv baselina na held-out sintetickim gradovima pokretanje: python -m tndp.experiments.bench_synth

import argparse
import time


from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import UNSERVED_FACTOR
from tndp.experiments.common import (evaluate_method, held_out_cities,
                                     load_policy, paired_cells, scales_for,
                                     write_table)
from tndp.rl.evaluate import decode, decode_sampling
from tndp.viz.methods import plot_synth
from tndp import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cities", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=None)
    # ime izlaza bez ekstenzije; ablacije i seed-ovi pisu u svoj fajl da ne prepisu glavnu tabelu
    ap.add_argument("--out", default="bench-synth")
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    a = args.alpha if args.alpha is not None else cfg["alpha_eval"]

    cities = held_out_cities(cfg, args.cities)
    scales = scales_for(cities)

    methods = {
        "random 200": lambda c: random_search(c, R, lo, hi, num_samples=200, alpha=a)[0],
        "greedy": lambda c: greedy_network(c, R, lo, hi, alpha=a)[0],
        "hill climbing": lambda c: hill_climb(c, R, lo, hi, alpha=a)[0],
        "RL greedy dekod": lambda c: decode(policy, c, R, lo, hi, a)[0],
        "RL sampling 32": lambda c: decode_sampling(policy, c, R, k=32, min_len=lo,
                                                    max_len=hi, alpha=a)[0],
    }

    stats, times = {}, {}
    for name, solve in methods.items():
        t0 = time.perf_counter()
        stats[name] = evaluate_method(solve, cities, scales, R, lo, hi, a)
        times[name] = (time.perf_counter() - t0) / len(cities)
        print(f"{name}: cilj {stats[name]['cilj'].mean():.3f} "
              f"({times[name]:.2f} s/grad)")

    ref = "greedy"
    lines = [
        f"# Held-out sintetika ({args.cities} gradova, n {cfg['n_range']}, "
        f"R={R}, alpha={a}, model {args.checkpoint})", "",
        "`cilj` = alpha * C_p_all/donja_granica + (1-alpha) * C_o/MST; manje je bolje.",
        f"Nepokrivena tražnja je već u C_p_all (nepokriven par se naplaćuje "
        f"{UNSERVED_FACTOR:g}x uličnim",
        "najkraćim vremenom), pa nema zasebne kazne. `C_p` je prosek samo nad opsluženim",
        "parovima i dat je radi poređenja sa literaturom, između metoda sa različitim",
        "`d_un` nije uporediv, za to služi `C_p_all`.", "",
        "± je standardna devijacija po gradovima. Δ i p su **uparene** razlike u `cilj`",
        f"u odnosu na `{ref}` (Wilcoxon, isti gradovi); Δ>0 znači bolje od reference.", "",
        "| metoda | cilj | Δ vs greedy | p | C_p_all | C_p | C_o | d_0 | d_un | s/grad |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        delta, p = paired_cells(s["cilj"], stats[ref]["cilj"], name == ref)
        lines.append(
            f"| {name} | {s['cilj'].mean():.3f} ± {s['cilj'].std(ddof=1):.3f} "
            f"| {delta} | {p} | {s['C_p_all'].mean():.2f} | {s['C_p'].mean():.2f} "
            f"| {s['C_o'].mean():.0f} | {s['d_0'].mean():.2f} "
            f"| {s['d_un'].mean():.3f} | {times[name]:.2f} |")

    write_table(f"{args.out}.md", lines)
    plot_synth({k: {m: float(v.mean()) for m, v in s.items()}
                for k, s in stats.items()}, RESULTS / f"{args.out}.png")


if __name__ == "__main__":
    main()
