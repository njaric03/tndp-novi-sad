# REINFORCE sa naučenim baseline-om (value glava), po Kool et al. šablonu.
# pokretanje: python -m tndp.rl.train --config configs/rl_smoke.yaml
# log ide u runs/<ime>/log.csv, checkpointi u runs/<ime>/{policy,best}.pt

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
    n_range=[15, 25], demand_mode="gravity", hidden=64, layers=3,
    eval_every=25, eval_cities=8, seed=0,
    # pool: fiksan skup trening gradova (kao Holliday), brže od generisanja
    # u letu i reproducibilnije. baseline: "value" (naučen) ili "greedy"
    # (self-critical, Kool et al.)
    pool_size=512, baseline="value",
    # alpha se uzorkuje po epizodi (kao Holliday), pa jedna politika pokriva
    # ceo Pareto front putnik/operater umesto samo alpha=0.5. alpha_fixed
    # zaključava vrednost, za ablaciju.
    alpha_fixed=None, alpha_eval=0.5,
    # best.pt se bira preko VIŠE tačaka fronta, ne samo alpha_eval: politika
    # koja pokriva ceo front ne sme da se selektuje u jednoj njegovoj tački.
    # alpha_eval ostaje tačka koja se loguje i prikazuje.
    val_alphas=[0.25, 0.5, 0.75],
    # standardizacija advantage-a unutar batch-a; REINFORCE sa terminalnom
    # nagradom je inače vrlo šumovit
    standardize_adv=True,
)


def make_pool(cfg, base_seed, count):
    return [generate_city(seed=base_seed + k, demand_mode=cfg["demand_mode"],
                          n_range=tuple(cfg["n_range"]))
            for k in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="pregazi seed iz configa (za varijansu po seed-u)")
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")))
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["name"] = f"{cfg['name']}-s{args.seed}"

    torch.manual_seed(cfg["seed"])
    policy = TndpPolicy(hidden=cfg["hidden"], layers=cfg["layers"])
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    pool = make_pool(cfg, 0, cfg["pool_size"])
    # fiksni validation set: gradovi koje trening nikad ne vidi
    val_cities = make_pool(cfg, 10_000, cfg["eval_cities"])
    a_eval = cfg["alpha_eval"]
    # alpha_eval je uvek među tačkama validacije, čak i ako nije u val_alphas
    val_alphas = sorted({float(a) for a in cfg["val_alphas"]} | {float(a_eval)})
    val_envs = {a: [TndpEnv(c, cfg["num_routes"], cfg["min_len"],
                            cfg["max_len"], a) for c in val_cities]
                for a in val_alphas}

    # random baseline na istim gradovima, po tački fronta. služi za poređenje
    # u logu, i kao skala pri izboru best.pt: cilj na alpha=0.25 i na 0.75 se
    # razlikuje i po apsolutnom nivou (~0.9 vs ~1.8), pa bi prost prosek
    # nagrada preko tačaka merio uglavnom gornji kraj fronta
    rand_reward = {}
    for a in val_alphas:
        rr = []
        for env in val_envs[a]:
            net, _ = random_search(env.city, cfg["num_routes"], cfg["min_len"],
                                   cfg["max_len"], num_samples=200, alpha=a)
            env.routes = net.routes
            rr.append(env.reward()[0])
        rand_reward[a] = float(np.mean(rr))

    out = Path("runs") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    log = csv.writer(open(out / "log.csv", "w", newline=""))
    log.writerow(["iter", "reward", "d_un", "entropy", "value_loss",
                  "val_reward", "sec"])

    rng = np.random.default_rng(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    best_val = -np.inf
    t0 = time.time()
    for it in range(1, cfg["iters"] + 1):
        rewards, d_uns, ents, vlosses = [], [], [], []
        logps, advs, vloss_terms = [], [], []
        opt.zero_grad()
        for _ in range(cfg["batch"]):
            city = pool[int(rng.integers(len(pool)))]
            # svaka epizoda dobija svoj kompromis putnik/operater; feature
            # alpha u modelu time postaje informativan i politika se na
            # evaluaciji može uslovljavati na bilo koju tačku fronta
            alpha = (cfg["alpha_fixed"] if cfg["alpha_fixed"] is not None
                     else float(rng.uniform(0.0, 1.0)))
            env = TndpEnv(city, cfg["num_routes"], cfg["min_len"],
                          cfg["max_len"], alpha)
            net, reward, res, logp, ent = rollout(policy, env, sample=True, gen=gen)
            # nekonačna nagrada bi kroz standardizaciju advantage-a postala
            # NaN i tiho otrovala sve težine; bolje pasti odmah i glasno
            if not np.isfinite(reward):
                raise RuntimeError(
                    f"nekonačna nagrada na {city.name}: cilj={-reward}, "
                    f"skale={env.scales}, linije={env.routes}")
            if cfg["baseline"] == "greedy":
                # self-critical: baseline je greedy dekodiranje iste politike
                with torch.no_grad():
                    _, greedy_r, _, _, _ = rollout(policy, env, sample=False)
                adv = reward - greedy_r
                vloss = torch.zeros(())
            else:
                # value glava predviđa nagradu iz početnog stanja
                env.reset()
                h0 = policy.encode(node_features(env), *edge_tensors(city))
                value = policy.state_value(h0)
                adv = reward - value.item()
                vloss = (value - reward) ** 2
            logps.append(logp)
            advs.append(adv)
            vloss_terms.append(vloss)
            rewards.append(reward)
            d_uns.append(res.d["d_un"])
            ents.append(ent)
            vlosses.append(float(vloss.detach()))

        adv_t = torch.tensor(advs, dtype=torch.float32)
        if cfg["standardize_adv"] and len(advs) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ent_t = torch.stack(ents)
        loss = (-(adv_t * torch.stack(logps)).mean()
                - cfg["entropy_coef"] * ent_t.mean()
                + cfg["value_coef"] * torch.stack(vloss_terms).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
        opt.step()

        val_reward = ""
        if it % cfg["eval_every"] == 0 or it == cfg["iters"]:
            by_alpha = {}
            for a in val_alphas:
                vr = []
                for env in val_envs[a]:
                    net, _ = decode(policy, env.city, cfg["num_routes"],
                                    cfg["min_len"], cfg["max_len"], a)
                    env.routes = net.routes
                    vr.append(env.reward()[0])
                by_alpha[a] = float(np.mean(vr))
            val_reward = by_alpha[a_eval]
            # skor za izbor best.pt: koliko je politika bolja od random
            # searcha, prosečeno preko tačaka fronta. odnos a ne razlika, da
            # tačke sa većim apsolutnim ciljem ne bi dominirale
            val_score = float(np.mean([rand_reward[a] / by_alpha[a]
                                       for a in val_alphas]))
            ckpt = {"cfg": cfg, "state_dict": policy.state_dict(), "iter": it,
                    "val_reward": val_reward, "val_by_alpha": by_alpha,
                    "val_score": val_score}
            torch.save(ckpt, out / "policy.pt")
            # najbolji na validaciji, ne poslednja iteracija: REINFORCE ume
            # da se pokvari pred kraj i onda se isporučuje gori model
            if val_score > best_val:
                best_val = val_score
                torch.save(ckpt, out / "best.pt")
            front = " ".join(f"a{a:g}:{by_alpha[a]:.2f}" for a in val_alphas)
            print(f"[{it}] reward {np.mean(rewards):.3f} | val {val_reward:.3f} "
                  f"| front {front} | vs random {val_score:.3f} "
                  f"(best {best_val:.3f}) | d_un {np.mean(d_uns):.2f} "
                  f"| ent {float(ent_t.mean().detach()):.2f}")
        log.writerow([it, np.mean(rewards), np.mean(d_uns),
                      float(ent_t.mean().detach()), np.mean(vlosses),
                      val_reward, round(time.time() - t0, 1)])

    print(f"gotovo za {(time.time() - t0) / 60:.1f} min, model u {out} "
          f"(policy.pt = poslednji, best.pt = najbolji na validaciji)")


if __name__ == "__main__":
    main()
