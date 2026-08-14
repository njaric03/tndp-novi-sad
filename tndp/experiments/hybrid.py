# Politika kao operator unutar metaheuristike, umesto samostalne metode

import argparse
import time

import numpy as np

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.core.assignment import assign, objective
from tndp.experiments.common import (fmt_p, held_out_cities, load_policy,
                                     paired_vs, scales_for)
from tndp.rl.evaluate import decode_sampling
from tndp.viz.style import color_for, save
from tndp import RESULTS

EVALS = 3000       # ukupan budzet evaluacija cilja po gradu
GRID = np.unique(np.round(np.logspace(0, np.log10(EVALS), 40)).astype(int))


# hill climbing daje trace samo u trenucima poboljsanja, ovde se prevodi na ravnomernu mrezu
def on_grid(trace, offset=0):
    ev = np.array([t[0] + offset for t in trace], dtype=float)
    ob = np.array([t[2] for t in trace], dtype=float)
    out = np.full(len(GRID), np.nan)
    for i, g in enumerate(GRID):
        prior = ev <= g
        if prior.any():
            out[i] = ob[prior][-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cities", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--evals", type=int, default=EVALS)
    args = ap.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    a = args.alpha if args.alpha is not None else cfg["alpha_eval"]
    k = args.samples

    cities = held_out_cities(cfg, args.cities)
    scales = scales_for(cities)

    # Startne tacke se NAPLACUJU. Slucajna je besplatna, RL sampling k kosta k evaluacija
    starts = {"slučajan start": "random", "greedy start": "greedy",
              f"RL sampling {k} start": "rl"}

    objs = {name: [] for name in starts}
    secs = {name: [] for name in starts}
    curves = {name: [] for name in starts}
    spent_evals = {name: [] for name in starts}
    # sama politika, bez penjanja, referenca "koliko pretraga dodaje"
    objs[f"RL sampling {k} (bez penjanja)"] = []
    secs[f"RL sampling {k} (bez penjanja)"] = []

    for ci, (city, sc) in enumerate(zip(cities, scales)):
        t0 = time.perf_counter()
        rl_net, _ = decode_sampling(policy, city, R, k=k, min_len=lo,
                                    max_len=hi, alpha=a)
        rl_secs = time.perf_counter() - t0
        assert rl_net.check(city, R, lo, hi) == [], rl_net.check(city, R, lo, hi)
        rl_obj = objective(assign(city, rl_net, compute_transfers=False), sc, a)
        objs[f"RL sampling {k} (bez penjanja)"].append(rl_obj)
        secs[f"RL sampling {k} (bez penjanja)"].append(rl_secs)

        for name, kind in starts.items():
            init, spent, extra_secs = kind, 0, 0.0
            if kind == "rl":
                init, spent, extra_secs = rl_net, k, rl_secs
            elif kind == "greedy":
                ge, t1 = [], time.perf_counter()
                init = greedy_network(city, R, lo, hi, alpha=a, evals=ge)[0]
                spent, extra_secs = ge[0], time.perf_counter() - t1
            trace = []
            t0 = time.perf_counter()
            net, obj = hill_climb(city, R, lo, hi, alpha=a, seed=ci,
                                  max_evals=max(1, args.evals - spent),
                                  init=init, trace=trace)
            dt = time.perf_counter() - t0 + extra_secs
            assert net.check(city, R, lo, hi) == [], net.check(city, R, lo, hi)
            objs[name].append(obj)
            secs[name].append(dt)
            curves[name].append(on_grid(trace, offset=spent))
            spent_evals[name].append(spent)
        print(f"  [{ci + 1}/{len(cities)}] {city.name}: "
              + ", ".join(f"{n} {objs[n][-1]:.3f}" for n in objs))

    objs = {n: np.array(v) for n, v in objs.items()}
    ref = "slučajan start"

    lines = [f"# Hibrid: politika kao startna tačka lokalne pretrage "
             f"(model {args.checkpoint})", "",
             f"{args.cities} held-out gradova, R={R}, alpha={a}. Ukupan budžet "
             f"je {args.evals} evaluacija cilja po gradu.",
             "Startna tačka se naplaćuje iz istog budžeta, kolona `start "
             "(ev.)` je koliko je potrošeno pre",
             "nego što je penjanje uopšte krenulo. Δ i p su uparene razlike u "
             f"odnosu na `{ref}` (Wilcoxon,",
             "isti gradovi); Δ>0 znači bolje.", "",
             "| start | start (ev.) | cilj | Δ vs slučajan | p | s/grad |",
             "|---|---|---|---|---|---|"]
    for name, v in objs.items():
        d, se, p = paired_vs(v, objs[ref])
        delta = "-" if name == ref else f"{d:+.3f} ± {se:.3f}"
        ev = f"{np.mean(spent_evals[name]):.0f}" if name in spent_evals else "-"
        lines.append(f"| {name} | {ev} | {v.mean():.3f} ± {v.std(ddof=1):.3f} | "
                     f"{delta} | {'-' if name == ref else fmt_p(p)} | "
                     f"{np.mean(secs[name]):.2f} |")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "hybrid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, cs in curves.items():
        stack = np.vstack(cs)
        # crta se samo tamo gde SVI gradovi imaju vrednost
        full = ~np.isnan(stack).any(axis=0)
        ax.plot(GRID[full], stack[:, full].mean(axis=0), label=name, lw=1.8,
                color=color_for("hibrid" if "RL" in name else name.split()[0]))
    ax.axhline(objs[f"RL sampling {k} (bez penjanja)"].mean(), ls="--", lw=1.2,
               color=color_for("RL sampling"), label=f"RL sampling {k} sam")
    ax.set_xscale("log")
    ax.set_xlabel("evaluacije cilja (log)")
    ax.set_ylabel("cilj (manje je bolje)")
    ax.set_title("Da li RL start ubrzava lokalnu pretragu")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("snimljeno u " + ", ".join(save(fig, RESULTS / "hybrid")))


if __name__ == "__main__":
    main()
