# zajednicko za skripte u experiments/: ucitavanje politike, uparena statistika i formatiranje tabela

import numpy as np
import torch
from scipy.stats import wilcoxon

from tndp.core.assignment import assign, cost_scales, objective
from tndp.rl.model import TndpPolicy
from tndp.core.synth import generate_city
from tndp import RESULTS

SEED_BASE = 20_000  # van trening poola (0..pool) i validacije (10k+)


# ucitaj checkpoint; podrazumevano best.pt (najbolji na validaciji) ako postoji, jer policy.pt je samo poslednja iteracija
def load_policy(path):
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["cfg"]
    # checkpointi napravljeni pre uvodjenja v2 featura nemaju kljuc; svi su v1
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


def evaluate_method(solve, cities, scales, num_routes, min_len, max_len, alpha):
    per_city = {k: [] for k in ("cilj", "C_p", "C_p_all", "C_o", "d_0", "d_un")}
    for city, sc in zip(cities, scales):
        net = solve(city)
        net.require_valid(city, num_routes, min_len, max_len)
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


def paired_vs(values, reference):
    d = reference - values          # >0 znaci da je metoda bolja od reference
    se = d.std(ddof=1) / np.sqrt(len(d))
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(values, reference).pvalue)
    return float(d.mean()), float(se), p


# "-" je rezervisano za referentni red, koji pozivaoci upisuju sami; ovde se
# zato i vrlo velike p vrednosti ispisuju kao broj, da se ta dva ne pobrkaju
def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    return ">0.999" if p > 0.999 else f"{p:.3f}"


# celije "Δ vs referenca" i "p" za jedan red tabele; referentni red dobija crtice
def paired_cells(values, reference, is_reference=False):
    if is_reference:
        return "-", "-"
    d, se, p = paired_vs(values, reference)
    return f"{d:+.3f} ± {se:.3f}", fmt_p(p)


# svaka skripta pise tabelu na isti nacin: jedan fajl u results/, pa poruka gde je
def write_table(name, lines):
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"snimljeno u {out}")
    return out


# Holm korekcija za vise poredjenja nad istim podacima
# jedna tabela testira 4-5 metoda protiv iste reference, pa je sirova p vrednost
# preoptimisticna: pri 5 testova i pragu 0.05 jedan lazno pozitivan je ocekivan
def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [0.0] * m
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * ps[i]))
        out[i] = running
    return out
