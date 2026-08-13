# REINFORCE sa naucenim baseline-om (value glava), po Kool et al

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from tndp.baselines.random_search import random_search
from tndp.rl.env import TndpEnv
from tndp.rl.evaluate import decode, decode_sampling, rollout
from tndp.rl.features import edge_tensors, node_features
from tndp.rl.model import TndpPolicy
from tndp.synth import generate_city

DEFAULTS = dict(
    name="bez-imena", iters=200, batch=8, lr=1e-4, entropy_coef=0.01,
    value_coef=0.5, grad_clip=1.0, num_routes=4, min_len=2, max_len=8,
    n_range=[15, 25], demand_mode="gravity", hidden=64, layers=3,
    eval_every=25, eval_cities=8, seed=0,
    # pool: fiksan skup trening gradova (kao Holliday), brze od generisanja u letu i reproducibilnije
    pool_size=512, baseline="value",
    # alpha se uzorkuje po epizodi (kao Holliday), pa jedna politika pokriva ceo Pareto front putnik/operater umesto samo
    alpha_fixed=None, alpha_eval=0.5,
    # best.pt se bira preko VISE tacaka fronta, ne samo alpha_eval: politika koja pokriva ceo front ne sme da se selektuje u
    val_alphas=[0.25, 0.5, 0.75],
    # standardizacija advantage-a unutar batch-a; REINFORCE sa terminalnom nagradom je inace vrlo sumovit
    standardize_adv=True,
    # skup ulaznih featura; "v2" dodaje prolaznost, koreness i bliskost i popravlja skaliranje stepena (vidi tndp/rl/model.py)
    features="v1",
    # koliko uzoraka koristi validacija. 1 je argmax, sto je i bilo ponasanje do
    # sada i zato ostaje podrazumevano: svi objavljeni modeli su tako birani.
    # Rezultati se izvestavaju pod uzorkovanjem 32, pa argmax meri drugu stvar.
    # Mereno na tri semena gravity-v1, poredak modela je isti kod oba dekodera
    # (Spearman +1.0), ali je nivo pomeren: argmax daje 2.237 tamo gde
    # uzorkovanje 32 daje 1.877, dok val_samples=8 daje 1.889.
    # Dakle argmax dobro RANGIRA, ali lose PROCENJUJE. Za izbor najbolje
    # iteracije unutar runa to je verovatno svejedno; za poredjenje sa
    # objavljenim brojevima nije.
    val_samples=1,
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
    parser.add_argument("--init", default=None,
                        help="topao start: težine iz gotovog checkpointa")
    parser.add_argument("--resume", action="store_true",
                        help="nastavi prekinut trening iz runs/<ime>/policy.pt")
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        # ime run-a je ime configa: configs/<ime>.yaml <-> runs/<ime>/
        cfg["name"] = Path(args.config).stem
        cfg.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")))
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["name"] = f"{cfg['name']}-s{args.seed}"

    torch.manual_seed(cfg["seed"])
    policy = TndpPolicy(hidden=cfg["hidden"], layers=cfg["layers"],
                        features=cfg["features"])
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    out = Path("runs") / cfg["name"]
    rng = np.random.default_rng(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    start_iter, best_val, t_offset = 1, -np.inf, 0.0

    # topao start: tezine gotovog treninga kao pocetna tacka
    if args.init:
        src = torch.load(args.init, map_location="cpu", weights_only=False)
        # skup featura mora da se poklopi, inace bi se tezine ulaznog sloja prenele na feature koji vise ne znace isto
        src_version = src["cfg"].get("features", "v1")
        if src_version != cfg["features"]:
            raise SystemExit(f"checkpoint je na featurima {src_version}, config traži "
                             f"{cfg['features']}, topao start nije moguć")
        policy.load_state_dict(src["state_dict"])
        print(f"topao start iz {args.init}: iteracija {src['iter']}, "
              f"vs random {src['val_score']:.3f}")

    # nastavak prekinutog treninga
    if args.resume:
        ck = torch.load(out / "policy.pt", map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt_state"])
        rng.bit_generator.state = ck["rng_state"]
        gen.set_state(ck["gen_state"])
        start_iter, best_val, t_offset = ck["iter"] + 1, ck["best_val"], ck["sec"]
        print(f"nastavak iz {out / 'policy.pt'}: iteracija {start_iter}, "
              f"best {best_val:.3f}")

    pool = make_pool(cfg, 0, cfg["pool_size"])
    # fiksni validation set: gradovi koje trening nikad ne vidi
    val_cities = make_pool(cfg, 10_000, cfg["eval_cities"])
    a_eval = cfg["alpha_eval"]
    # alpha_eval je uvek medju tackama validacije, cak i ako nije u val_alphas
    val_alphas = sorted({float(a) for a in cfg["val_alphas"]} | {float(a_eval)})
    val_envs = {a: [TndpEnv(c, cfg["num_routes"], cfg["min_len"],
                            cfg["max_len"], a) for c in val_cities]
                for a in val_alphas}

    # random baseline na istim gradovima, po tacki fronta
    rand_reward = {}
    for a in val_alphas:
        rr = []
        for env in val_envs[a]:
            net, _ = random_search(env.city, cfg["num_routes"], cfg["min_len"],
                                   cfg["max_len"], num_samples=200, alpha=a)
            env.routes = net.routes
            rr.append(env.reward()[0])
        rand_reward[a] = float(np.mean(rr))

    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.csv", "a" if args.resume else "w", newline="")
    log = csv.writer(log_f)
    if not args.resume:
        log.writerow(["iter", "reward", "d_un", "entropy", "value_loss",
                      "val_reward", "sec"])

    t0 = time.time()
    for it in range(start_iter, cfg["iters"] + 1):
        rewards, d_uns, ents, vlosses = [], [], [], []
        logps, advs, vloss_terms = [], [], []
        opt.zero_grad()
        for _ in range(cfg["batch"]):
            city = pool[int(rng.integers(len(pool)))]
            # svaka epizoda dobija svoj alpha, pa jedna politika pokriva ceo Pareto front
            alpha = (cfg["alpha_fixed"] if cfg["alpha_fixed"] is not None
                     else float(rng.uniform(0.0, 1.0)))
            env = TndpEnv(city, cfg["num_routes"], cfg["min_len"],
                          cfg["max_len"], alpha)
            net, reward, res, logp, ent = rollout(policy, env, sample=True, gen=gen)
            # nekonacna nagrada bi kroz standardizaciju advantage-a postala NaN i tiho otrovala sve tezine; bolje pasti odmah i glasno
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
                # value glava predvidja nagradu iz pocetnog stanja
                env.reset()
                h0 = policy.encode(node_features(env, policy.features), *edge_tensors(city))
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
                    if cfg["val_samples"] > 1:
                        net, _ = decode_sampling(
                            policy, env.city, cfg["num_routes"],
                            k=cfg["val_samples"], min_len=cfg["min_len"],
                            max_len=cfg["max_len"], alpha=a)
                    else:
                        net, _ = decode(policy, env.city, cfg["num_routes"],
                                        cfg["min_len"], cfg["max_len"], a)
                    env.routes = net.routes
                    vr.append(env.reward()[0])
                by_alpha[a] = float(np.mean(vr))
            val_reward = by_alpha[a_eval]
            # skor za izbor best.pt: koliko je politika bolja od random searcha, proseceno preko tacaka fronta
            val_score = float(np.mean([rand_reward[a] / by_alpha[a]
                                       for a in val_alphas]))
            # najbolji na validaciji, ne poslednja iteracija: REINFORCE ume da se pokvari pred kraj i onda se isporucuje gori model
            is_best = val_score > best_val
            best_val = max(best_val, val_score)
            # optimizer i RNG stanja idu u checkpoint da bi --resume mogao da nastavi tacno odatle; sec je ukupno vreme svih deonica
            ckpt = {"cfg": cfg, "state_dict": policy.state_dict(), "iter": it,
                    "val_reward": val_reward, "val_by_alpha": by_alpha,
                    "val_score": val_score, "best_val": best_val,
                    "opt_state": opt.state_dict(),
                    "rng_state": rng.bit_generator.state,
                    "gen_state": gen.get_state(),
                    "sec": time.time() - t0 + t_offset}
            torch.save(ckpt, out / "policy.pt")
            if is_best:
                torch.save(ckpt, out / "best.pt")
            front = " ".join(f"a{a:g}:{by_alpha[a]:.2f}" for a in val_alphas)
            print(f"[{it}] reward {np.mean(rewards):.3f} | val {val_reward:.3f} "
                  f"| front {front} | vs random {val_score:.3f} "
                  f"(best {best_val:.3f}) | d_un {np.mean(d_uns):.2f} "
                  f"| ent {float(ent_t.mean().detach()):.2f}")
        log.writerow([it, np.mean(rewards), np.mean(d_uns),
                      float(ent_t.mean().detach()), np.mean(vlosses),
                      val_reward, round(time.time() - t0 + t_offset, 1)])
        # bez flush-a poslednjih ~60 redova ostane u baferu ako proces bude ubijen spolja; tako je pao runs/novisad-r19
        log_f.flush()

    print(f"gotovo za {(time.time() - t0 + t_offset) / 60:.1f} min, model u "
          f"{out} (policy.pt = poslednji, best.pt = najbolji na validaciji)")


if __name__ == "__main__":
    main()
