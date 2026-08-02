# poređenje načina dekodiranja iste trenirane politike: greedy, sampling i
# MCTS (frontier add-on). manji broj gradova jer je MCTS spor.
# pokretanje: python -m tndp.experiments.bench_decoders runs/gravity-v1/policy.pt

import sys
import time
from pathlib import Path

import numpy as np
import torch

from tndp.core.assignment import assign, cost_scales, objective
from tndp.rl.evaluate import decode, decode_sampling
from tndp.rl.mcts import mcts_decode
from tndp.rl.model import TndpPolicy
from tndp.synth.generator import generate_city

NUM_CITIES = 10
SEED_BASE = 20_000


def main():
    ckpt = torch.load(sys.argv[1], weights_only=False)
    cfg = ckpt["cfg"]
    policy = TndpPolicy(hidden=cfg["hidden"], layers=cfg["layers"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    R, lo, hi, a = cfg["num_routes"], cfg["min_len"], cfg["max_len"], cfg["alpha"]

    cities = [generate_city(seed=SEED_BASE + k, demand_mode=cfg["demand_mode"],
                            n_range=tuple(cfg["n_range"])) for k in range(NUM_CITIES)]
    scales = [cost_scales(c, R, hi) for c in cities]

    decoders = {
        "greedy dekod": lambda c: decode(policy, c, R, lo, hi, a),
        "sampling 32": lambda c: decode_sampling(policy, c, R, k=32, min_len=lo,
                                                 max_len=hi, alpha=a),
        "MCTS 30": lambda c: mcts_decode(policy, c, R, lo, hi, a, sims=30),
    }

    lines = [f"# Poređenje dekodera ({NUM_CITIES} gradova, ista politika {sys.argv[1]})",
             "", "| dekoder | cilj | C_p (min) | C_o (min) | d_un | sec/grad |",
             "|---|---|---|---|---|---|"]
    for name, dec in decoders.items():
        objs, cps, cos, duns = [], [], [], []
        t0 = time.perf_counter()
        for c, sc in zip(cities, scales):
            _, res = dec(c)
            objs.append(objective(res, sc, a))
            cps.append(res.C_p)
            cos.append(res.C_o)
            duns.append(res.d["d_un"])
        dt = (time.perf_counter() - t0) / NUM_CITIES
        lines.append(f"| {name} | {np.mean(objs):.3f} | {np.mean(cps):.2f} | "
                     f"{np.mean(cos):.0f} | {np.mean(duns):.3f} | {dt:.1f} |")
        print(lines[-1])

    out = Path(__file__).parent.parent.parent / "results" / "bench_decoders.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"snimljeno u {out}")


if __name__ == "__main__":
    main()
