# poredjenje nacina dekodiranja iste trenirane politike: greedy, sampling i MCTS. manji broj gradova jer je MCTS spor

import argparse
import time


from tndp.experiments.common import (evaluate_method, held_out_cities,
                                     load_policy, paired_cells, scales_for,
                                     write_table)
from tndp.rl.evaluate import decode, decode_sampling
from tndp.rl.mcts import mcts_decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cities", type=int, default=20)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    a = args.alpha if args.alpha is not None else cfg["alpha_eval"]
    cities = held_out_cities(cfg, args.cities)
    scales = scales_for(cities)

    decoders = {
        "greedy dekod": lambda c: decode(policy, c, R, lo, hi, a)[0],
        f"sampling {args.k}": lambda c: decode_sampling(policy, c, R, k=args.k,
                                                        min_len=lo, max_len=hi,
                                                        alpha=a)[0],
        f"MCTS {args.sims}": lambda c: mcts_decode(policy, c, R, lo, hi, a,
                                                   sims=args.sims)[0],
    }

    stats, times = {}, {}
    for name, dec in decoders.items():
        t0 = time.perf_counter()
        stats[name] = evaluate_method(dec, cities, scales, R, lo, hi, a)
        times[name] = (time.perf_counter() - t0) / len(cities)
        print(f"{name}: cilj {stats[name]['cilj'].mean():.3f} ({times[name]:.2f} s/grad)")

    ref = "greedy dekod"
    lines = [f"# Poređenje dekodera ({args.cities} gradova, alpha={a}, "
             f"ista politika {args.checkpoint})", "",
             "Δ i p su uparene razlike u odnosu na greedy dekodiranje "
             "(Wilcoxon, isti gradovi).", "",
             "| dekoder | cilj | Δ vs greedy | p | C_p_all | C_o | d_un | s/grad |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in stats.items():
        delta, p = paired_cells(s["cilj"], stats[ref]["cilj"], name == ref)
        lines.append(f"| {name} | {s['cilj'].mean():.3f} ± {s['cilj'].std(ddof=1):.3f} "
                     f"| {delta} | {p} | {s['C_p_all'].mean():.2f} "
                     f"| {s['C_o'].mean():.0f} | {s['d_un'].mean():.3f} "
                     f"| {times[name]:.2f} |")

    write_table("bench-decoders.md", lines)


if __name__ == "__main__":
    main()
