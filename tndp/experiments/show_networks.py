# nacrtaj mreze koje razne metode grade na istom held-out gradu
# pokretanje: python -m tndp.experiments.show_networks runs/gravity-v1/policy.pt

import sys
from pathlib import Path

import torch

from tndp.baselines.greedy import greedy_network
from tndp.baselines.random_search import random_search
from tndp.rl.evaluate import decode_sampling
from tndp.rl.model import TndpPolicy
from tndp.synth.generator import generate_city
from tndp.viz.maps import compare_networks

SEED = 20_000  # isti grad kao prvi u bench_synth


def main():
    ckpt = torch.load(sys.argv[1], weights_only=False)
    cfg = ckpt["cfg"]
    policy = TndpPolicy(hidden=cfg["hidden"], layers=cfg["layers"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    R, lo, hi, a = cfg["num_routes"], cfg["min_len"], cfg["max_len"], cfg["alpha"]

    city = generate_city(seed=SEED, demand_mode=cfg["demand_mode"],
                         n_range=tuple(cfg["n_range"]))
    nets = {
        "random (najbolja od 200)": random_search(city, R, lo, hi, num_samples=200)[0],
        "greedy": greedy_network(city, R, lo, hi, alpha=a)[0],
        "RL (sampling 32)": decode_sampling(policy, city, R, k=32, min_len=lo,
                                            max_len=hi, alpha=a)[0],
    }
    out = Path(__file__).parent.parent.parent / "results" / "networks.png"
    compare_networks(city, nets, out, alpha=a)
    print(f"snimljeno u {out}")


if __name__ == "__main__":
    main()
