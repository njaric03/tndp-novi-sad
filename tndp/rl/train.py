# REINFORCE sa naučenim baseline-om (value glava), po Kool et al. šablonu.
# pokretanje: python -m tndp.rl.train --config configs/rl_smoke.yaml
# log ide u runs/<ime>/log.csv, checkpoint u runs/<ime>/policy.pt

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from tndp.baselines.random_search import random_search
from tndp.rl.env import TndpEnv
from tndp.rl.evaluate import decode, rollout
from tndp.rl.model import TndpPolicy, edge_tensors, node_features
from tndp.synth.generator import generate_city

DEFAULTS = dict(
    name="smoke", iters=200, batch=8, lr=1e-4, entropy_coef=0.01,
    value_coef=0.5, grad_clip=1.0, num_routes=4, min_len=2, max_len=8,
    alpha=0.5, n_range=[15, 25], demand_mode="gravity", hidden=64, layers=3,
    eval_every=25, eval_cities=8, seed=0,
    # pool: fiksan skup trening gradova (kao Holliday), brže od generisanja
    # u letu i reproducibilnije. baseline: "value" (naučen) ili "greedy"
    # (self-critical, Kool et al.)
    pool_size=512, baseline="value",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")))

    torch.manual_seed(cfg["seed"])
    policy = TndpPolicy(hidden=cfg["hidden"], layers=cfg["layers"])
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    pool = [generate_city(seed=k, demand_mode=cfg["demand_mode"],
                          n_range=tuple(cfg["n_range"]))
            for k in range(cfg["pool_size"])]

    # fiksni validation set: gradovi koje trening nikad ne vidi
    val_cities = [generate_city(seed=10_000 + k, demand_mode=cfg["demand_mode"],
                                n_range=tuple(cfg["n_range"]))
                  for k in range(cfg["eval_cities"])]
    val_envs = [TndpEnv(c, cfg["num_routes"], cfg["min_len"], cfg["max_len"],
                        cfg["alpha"]) for c in val_cities]
    # random baseline na istim gradovima, za poređenje u logu
    rand_rewards = []
    for env in val_envs:
        net, _ = random_search(env.city, cfg["num_routes"], cfg["min_len"],
                               cfg["max_len"], num_samples=200, alpha=cfg["alpha"])
        env.routes = net.routes
        rand_rewards.append(env.reward()[0])
    rand_reward = float(np.mean(rand_rewards))

    out = Path("runs") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    log = csv.writer(open(out / "log.csv", "w", newline=""))
    log.writerow(["iter", "reward", "d_un", "entropy", "value_loss",
                  "val_reward", "sec"])

    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    for it in range(1, cfg["iters"] + 1):
        batch_loss, rewards, d_uns, ents, vlosses = 0.0, [], [], [], []
        opt.zero_grad()
        for _ in range(cfg["batch"]):
            city = pool[int(rng.integers(len(pool)))]
            env = TndpEnv(city, cfg["num_routes"], cfg["min_len"],
                          cfg["max_len"], cfg["alpha"])
            net, reward, res, logp, ent = rollout(policy, env, sample=True)
            if cfg["baseline"] == "greedy":
                # self-critical: baseline je greedy dekodiranje iste politike
                with torch.no_grad():
                    _, greedy_r, _, _, _ = rollout(policy, env, sample=False)
                adv = reward - greedy_r
                vloss = torch.tensor(0.0)
            else:
                # value glava predviđa nagradu iz početnog stanja
                env.reset()
                h0 = policy.encode(node_features(env), *edge_tensors(city))
                value = policy.state_value(h0)
                adv = reward - value.item()
                vloss = (value - reward) ** 2
            loss = (-adv * logp - cfg["entropy_coef"] * ent
                    + cfg["value_coef"] * vloss) / cfg["batch"]
            loss.backward()
            batch_loss += float(loss.detach())
            rewards.append(reward)
            d_uns.append(res.d["d_un"])
            ents.append(float(ent))
            vlosses.append(float(vloss))
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
        opt.step()

        val_reward = ""
        if it % cfg["eval_every"] == 0 or it == cfg["iters"]:
            vr = []
            for env in val_envs:
                net, _ = decode(policy, env.city, cfg["num_routes"],
                                cfg["min_len"], cfg["max_len"], cfg["alpha"])
                env.routes = net.routes
                vr.append(env.reward()[0])
            val_reward = float(np.mean(vr))
            torch.save({"cfg": cfg, "state_dict": policy.state_dict()},
                       out / "policy.pt")
            print(f"[{it}] reward {np.mean(rewards):.3f} | val {val_reward:.3f} "
                  f"| random {rand_reward:.3f} | d_un {np.mean(d_uns):.2f} "
                  f"| ent {np.mean(ents):.2f}")
        log.writerow([it, np.mean(rewards), np.mean(d_uns), np.mean(ents),
                      np.mean(vlosses), val_reward, round(time.time() - t0, 1)])

    print(f"gotovo za {(time.time() - t0) / 60:.1f} min, model u {out}")


if __name__ == "__main__":
    main()
