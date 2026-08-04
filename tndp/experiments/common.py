# zajedničko za skripte u experiments/: učitavanje politike, uparena
# statistika i formatiranje tabela

import numpy as np
import torch
from scipy.stats import wilcoxon

from tndp.core.assignment import assign, cost_scales, objective
from tndp.rl.model import TndpPolicy
from tndp.synth import generate_city

SEED_BASE = 20_000  # van trening poola (0..pool) i validacije (10k+)


# učitaj checkpoint; podrazumevano best.pt (najbolji na validaciji) ako
# postoji, jer policy.pt je samo poslednja iteracija
def load_policy(path):
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["cfg"]
    # checkpointi napravljeni pre uvođenja v2 featura nemaju ključ; svi su v1
    cfg.setdefault("features", "v1")
    policy = TndpPolicy(hidden=cfg["hidden"], layers=cfg["layers"],
                        features=cfg["features"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    # stariji configi su imali fiksni "alpha"; noviji "alpha_eval"
    cfg.setdefault("alpha_eval", cfg.get("alpha", 0.5))
    return policy, cfg


def held_out_cities(cfg, count):
    return [generate_city(seed=SEED_BASE + k, demand_mode=cfg["demand_mode"],
                          n_range=tuple(cfg["n_range"])) for k in range(count)]


# pusti jednu metodu preko svih gradova. VALIDIRA izlaz svake metode —
# ranije nijedan eksperiment nije proveravao da mreža poštuje ograničenja,
# pa bi duplirane ili prekratke linije prošle nezapaženo.
def evaluate_method(solve, cities, scales, num_routes, min_len, max_len, alpha):
    per_city = {k: [] for k in ("cilj", "C_p", "C_p_all", "C_o", "d_0", "d_un")}
    for city, sc in zip(cities, scales):
        net = solve(city)
        problems = net.check(city, num_routes, min_len, max_len)
        if problems:
            raise AssertionError(f"nevalidna mreža na {city.name}: {problems}")
        res = assign(city, net)
        per_city["cilj"].append(objective(res, sc, alpha))
        per_city["C_p"].append(res.C_p)
        per_city["C_p_all"].append(res.C_p_all)
        per_city["C_o"].append(res.C_o)
        per_city["d_0"].append(res.d["d_0"])
        per_city["d_un"].append(res.d["d_un"])
    return {k: np.array(v) for k, v in per_city.items()}


def scales_for(cities):
    return [cost_scales(c) for c in cities]


# uparena razlika u odnosu na referentnu metodu. gradovi se međusobno
# razlikuju po težini mnogo više nego metode, pa nespareno poređenje troši
# većinu osetljivosti ni na šta.
def paired_vs(values, reference):
    d = reference - values          # >0 znači da je metoda bolja od reference
    se = d.std(ddof=1) / np.sqrt(len(d))
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(values, reference).pvalue)
    return float(d.mean()), float(se), p


def fmt_p(p):
    return "—" if p >= 0.999 else ("<0.001" if p < 0.001 else f"{p:.3f}")
