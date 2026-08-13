# Pareto front putnik/operater: sweep po alpha

import argparse
from pathlib import Path

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.experiments.common import (evaluate_method, held_out_cities,
                                     load_policy, scales_for)
from tndp.rl.evaluate import decode_sampling
from tndp.viz import paper

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
            rows.append(f"| {a} | {name} | {s['cilj'].mean():.3f} "
                        f"| {s['C_p_all'].mean():.2f} | {s['C_o'].mean():.0f} "
                        f"| {s['d_un'].mean():.3f} |")
            print(rows[-1])

    results = Path(__file__).parent.parent.parent / "results"
    header = [f"# Pareto front ({args.cities} held-out gradova, R={R}, "
              f"model {args.checkpoint})", "",
              "Ista trenirana politika je puštena na svaku vrednost alpha "
              "(alpha je feature čvora i uzorkuje se tokom treninga);",
              "baselines se za svaku tačku pokreću iznova.", ""]
    (results / "pareto.md").write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    # sliku crta viz.paper iz ove iste tabele, da se figura u radu i tabela
    # ne mogu razici
    print(f"snimljeno u {results / 'pareto.md'}")
    print("->", *paper.pareto(results / "slika-pareto"))


if __name__ == "__main__":
    main()
