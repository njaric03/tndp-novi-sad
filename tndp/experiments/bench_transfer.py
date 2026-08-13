# Transfer: politika trenirana na sintetici pusta se na benchmark instance iz literature, bez ikakvog dotreniravanja

import argparse
import time
from pathlib import Path

import numpy as np

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign, cost_scales, objective
from tndp.core.io import load_benchmark_city
from tndp.experiments.common import load_policy
from tndp.rl.evaluate import decode, decode_sampling

DATA = Path(__file__).parent.parent.parent / "data" / "benchmarks"

# standardni parametri instanci (Mumford 2013, tabela u data/benchmarks/Mumford)
INSTANCES = {
    "Mandl1":   ("Mandl/Mandl1",     6,  2,  8),
    "Mumford0": ("Mumford/Mumford0", 12, 2, 15),
    "Mumford1": ("Mumford/Mumford1", 15, 10, 30),
    "Mumford2": ("Mumford/Mumford2", 56, 10, 22),
    "Mumford3": ("Mumford/Mumford3", 60, 12, 25),
}
DEFAULT = ["Mandl1", "Mumford0", "Mumford1"]
GREEDY_MAX_N = 40  # iznad toga je konstruktivni greedy prespor (O(R*n^2) assign-a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--instances", nargs="+", default=DEFAULT, choices=list(INSTANCES))
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", default="bench-transfer")
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    a = args.alpha if args.alpha is not None else cfg["alpha_eval"]

    lines = [f"# Transfer na benchmark instance (model {args.checkpoint}, alpha={a})", "",
             f"Politika je trenirana na sintetici sa n {cfg['n_range']}, "
             f"R={cfg['num_routes']}, dužina linije [{cfg['min_len']}, {cfg['max_len']}].",
             "Instance se puštaju sa svojim standardnim parametrima, dakle van te",
             "distribucije, kolona `van distr.` kaže koliko.", "",
             "| instanca | n | R | metoda | cilj | C_p_all | C_p | C_o | d_un | s |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    for key in args.instances:
        rel, R, lo, hi = INSTANCES[key]
        city = load_benchmark_city(DATA / rel)
        assert city.validate() == [], city.validate()
        scales = cost_scales(city)

        methods = {
            "random 200": lambda c: random_search(c, R, lo, hi, num_samples=200, alpha=a)[0],
            "hill climbing": lambda c: hill_climb(c, R, lo, hi, alpha=a)[0],
            "RL greedy dekod": lambda c: decode(policy, c, R, lo, hi, a)[0],
            f"RL sampling {args.samples}": lambda c: decode_sampling(
                policy, c, R, k=args.samples, min_len=lo, max_len=hi, alpha=a)[0],
        }
        if city.n <= GREEDY_MAX_N:
            methods["greedy"] = lambda c: greedy_network(c, R, lo, hi, alpha=a)[0]

        print(f"\n== {key} (n={city.n}, R={R}, dužina [{lo},{hi}]) ==")
        for name, solve in methods.items():
            t0 = time.perf_counter()
            net = solve(city)
            dt = time.perf_counter() - t0
            problems = net.check(city, R, lo, hi)
            if problems:
                print(f"  {name}: NEVALIDNO {problems[:2]}")
                continue
            res = assign(city, net)
            row = (f"| {key} | {city.n} | {R} | {name} | "
                   f"{objective(res, scales, a):.3f} | {res.C_p_all:.2f} | "
                   f"{res.C_p:.2f} | {res.C_o:.0f} | {res.d['d_un']:.3f} | {dt:.1f} |")
            lines.append(row)
            print("  " + row)

    out = Path(__file__).parent.parent.parent / "results" / f"{args.out}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsnimljeno u {out}")


if __name__ == "__main__":
    main()
