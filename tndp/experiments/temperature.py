# Temperatura pri uzorkovanju: best-of-k trosi uzorke na raznovrsnost, pa optimum nije na T=1

import argparse
import time

import numpy as np

from tndp.core.assignment import assign, cost_scales, objective
from tndp.core.io import load_benchmark_city
from tndp.experiments.bench_transfer import INSTANCES
from tndp.experiments.common import (fmt_p, held_out_cities, load_policy,
                                     paired_vs, scales_for, write_table)
from tndp.rl.evaluate import decode_sampling
from tndp.viz.style import save
from tndp import BENCHMARKS, RESULTS

TEMPS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cities", type=int, default=20)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=None)
    # instance iz literature umesto sintetike: tamo je politika van raspodele, pa
    # raznovrsnost uzorkovanja vredi drugacije nego na gradovima kakve je videla
    ap.add_argument("--instances", nargs="*", default=None, choices=list(INSTANCES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    a = args.alpha if args.alpha is not None else cfg["alpha_eval"]
    k = args.samples

    if args.instances:
        # svaka instanca nosi svoj R i granice duzine, ne one iz treninga
        cities, param = [], {}
        for ime in args.instances:
            put, iR, ilo, ihi = INSTANCES[ime]
            c = load_benchmark_city(BENCHMARKS / put)
            c.name = ime
            cities.append(c)
            param[ime] = (iR, ilo, ihi)
        scales = [cost_scales(c) for c in cities]
    else:
        param = None
        cities = held_out_cities(cfg, args.cities)
        scales = scales_for(cities)

    objs, secs, d_uns = {}, {}, {}
    for t in TEMPS:
        vals, t0 = [], time.perf_counter()
        for city, sc in zip(cities, scales):
            cR, clo, chi = param[city.name] if param else (R, lo, hi)
            net, res = decode_sampling(policy, city, cR, k=k, min_len=clo,
                                       max_len=chi, alpha=a, temps=[t])
            # van raspodele maskiranje ne garantuje validnost, pa se ne obara ceo run
            res = assign(city, net, compute_transfers=False)
            vals.append((objective(res, sc, a), res.d["d_un"]))
        objs[t] = np.array([v[0] for v in vals])
        d_uns[t] = np.array([v[1] for v in vals])
        secs[t] = (time.perf_counter() - t0) / len(cities)
        print(f"  T={t:.1f}  cilj {objs[t].mean():.3f} ± {objs[t].std(ddof=1):.3f}"
              f"  d_un {d_uns[t].mean():.3f}  {secs[t]:.2f} s/grad")

    ref = 1.0
    best = min(TEMPS, key=lambda t: objs[t].mean())
    lines = [f"# Temperatura uzorkovanja (model {args.checkpoint})", "",
             f"{', '.join(c.name for c in cities) if param else str(args.cities) + ' held-out gradova'}, alpha={a}, "
             f"uzorkovanje {k}. Logiti se dele sa `T` samo pri dekodiranju,",
             "trenirana politika se ne dira. `T<1` izoštrava raspodelu, `T>1` je širi.",
             "",
             f"Δ i p su uparene razlike u odnosu na `T={ref:.1f}` (Wilcoxon, isti "
             "gradovi); Δ>0 znači bolje.", "",
             "| T | cilj | Δ vs T=1 | p | d_un | s/grad |", "|---|---|---|---|---|---|"]
    for t in TEMPS:
        d, se, p = paired_vs(objs[t], objs[ref])
        delta = "-" if t == ref else f"{d:+.3f} ± {se:.3f}"
        mark = " **<-**" if t == best else ""
        lines.append(f"| {t:.1f}{mark} | {objs[t].mean():.3f} ± "
                     f"{objs[t].std(ddof=1):.3f} | {delta} | "
                     f"{'-' if t == ref else fmt_p(p)} | {d_uns[t].mean():.3f} | "
                     f"{secs[t]:.2f} |")
    lines += ["", f"Najbolja je **T = {best:.1f}** ({objs[best].mean():.3f} naspram "
                  f"{objs[ref].mean():.3f} na T=1). Temperatura ne menja cenu "
                  "dekodiranja,", "pa je to poboljšanje bez dodatnog računa."]

    write_table(f"{args.out or 'temperature'}.md", lines)
    print("\n" + "\n".join(lines))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    m = np.array([objs[t].mean() for t in TEMPS])
    se = np.array([objs[t].std(ddof=1) / np.sqrt(len(cities)) for t in TEMPS])
    ax.errorbar(TEMPS, m, yerr=se, marker="o", lw=1.8, capsize=3)
    ax.axvline(1.0, ls="--", lw=1.0, color="gray")
    ax.set_xlabel("temperatura uzorkovanja")
    ax.set_ylabel("cilj (manje je bolje)")
    ax.set_title(f"Temperatura pri best-of-{k} dekodiranju")
    ax.grid(alpha=0.3)
    print("snimljeno u " + ", ".join(save(fig, RESULTS / (args.out or "temperature"))))


if __name__ == "__main__":
    main()
