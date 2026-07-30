# RL politika protiv baselina na held-out sintetičkim gradovima
# pokretanje: python -m tndp.experiments.bench_synth runs/gravity-v1/policy.pt

import sys
from pathlib import Path

import numpy as np
import torch

from tndp.baselines.greedy import greedy_network
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign, cost_scales, objective
from tndp.rl.evaluate import decode, decode_sampling
from tndp.rl.model import TndpPolicy
from tndp.synth.generator import generate_city
from tndp.viz.bench import plot_synth

NUM_CITIES = 20
SEED_BASE = 20_000  # van trening poola (0..pool) i validacije (10k+)


def main():
    ckpt = torch.load(sys.argv[1], weights_only=False)
    cfg = ckpt["cfg"]
    policy = TndpPolicy(hidden=cfg["hidden"], layers=cfg["layers"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]

    cities = [generate_city(seed=SEED_BASE + k, demand_mode=cfg["demand_mode"],
                            n_range=tuple(cfg["n_range"]))
              for k in range(NUM_CITIES)]

    methods = {
        "random najbolja od 200": lambda c: random_search(c, R, lo, hi, num_samples=200)[0],
        "greedy": lambda c: greedy_network(c, R, lo, hi, alpha=cfg["alpha"])[0],
        "RL greedy dekod": lambda c: decode(policy, c, R, lo, hi, cfg["alpha"])[0],
        "RL sampling 32": lambda c: decode_sampling(policy, c, R, k=32, min_len=lo,
                                                    max_len=hi, alpha=cfg["alpha"])[0],
    }

    scales = [cost_scales(c, R, hi) for c in cities]

    stats = {}
    for name, solve in methods.items():
        objs, cps, cos, d0s, duns = [], [], [], [], []
        for c, sc in zip(cities, scales):
            res = assign(c, solve(c))
            objs.append(objective(res, sc, cfg["alpha"]))
            cps.append(res.C_p)
            cos.append(res.C_o)
            d0s.append(res.d["d_0"])
            duns.append(res.d["d_un"])
        stats[name] = {k: float(np.mean(v)) for k, v in
                       dict(cilj=objs, C_p=cps, C_o=cos, d_0=d0s, d_un=duns).items()}
        print(f"{name}: cilj {stats[name]['cilj']:.3f}")

    lines = [f"# Held-out sintetika ({NUM_CITIES} gradova, n {cfg['n_range']}, "
             f"R={R}, alpha={cfg['alpha']}, model {sys.argv[1]})", "",
             "cilj je normalizovani kombinovani cost plus kazna za nepokriven "
             "demand, tačno ono što RL optimizuje; manje je bolje. sve metode "
             "se porede istim skalarom.", "",
             "| metoda | cilj | C_p (min) | C_o (min) | d_0 | d_un |",
             "|---|---|---|---|---|---|"]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['cilj']:.3f} | {s['C_p']:.2f} | {s['C_o']:.0f} "
                     f"| {s['d_0']:.2f} | {s['d_un']:.3f} |")

    results = Path(__file__).parent.parent.parent / "results"
    (results / "bench_synth.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_synth(stats, results / "bench_synth.png")
    print(f"snimljeno u {results / 'bench_synth.md'} i .png")


if __name__ == "__main__":
    main()
