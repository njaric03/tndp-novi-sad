# Provere invarijanti na kojima počiva funkcija cilja, plus osetljivost na
# dve konstante koje u njoj ostaju. Ovo NIJE test suite — svrha je da brojevi
# koji idu u rad imaju proverljivo poreklo, i da se osetljivost objavi umesto
# da se ćuti o njoj.
#
# Za stanje pre popravki i za merenja koja su ih motivisala videti
# docs/metodoloska-procena.md (brojevi su iz commita 9e23840).
#
# pokretanje: python -m tools.metodoloske_provere


import numpy as np

import tndp.core.assignment as A
from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign, cost_scales, objective
from tndp.core.io import load_benchmark_city
from tndp.core.network import TransitNetwork
from tndp.synth.generator import generate_city

SEED_BASE, NUM_CITIES = 20_000, 12
R, MIN_LEN, MAX_LEN, ALPHA, N_RANGE = 4, 2, 8, 0.5, (15, 30)


def cities(k=NUM_CITIES):
    return [generate_city(seed=SEED_BASE + i, demand_mode="gravity",
                          n_range=N_RANGE) for i in range(k)]


def hdr(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def greedy_results(cs, alpha=ALPHA):
    return [assign(c, greedy_network(c, R, MIN_LEN, MAX_LEN, alpha=alpha)[0])
            for c in cs]


# --- 1. da li alpha zaista deli uticaj na pola ------------------------------
# kriterijum je rasipanje člana preko kandidat-rešenja, ne njegov nivo:
# alpha balansira to koliko svaki član MENJA cilj kad se mreža promeni
def check_balance(cs):
    hdr("1. balans putničkog i operaterskog člana (cilj ~1:1)")
    cp_sd, co_sd = [], []
    for c in cs[:8]:
        sc = cost_scales(c)
        rng = np.random.default_rng(0)
        cp, co = [], []
        for _ in range(120):
            net = random_search(c, R, MIN_LEN, MAX_LEN, num_samples=1,
                                alpha=ALPHA, seed=int(rng.integers(1 << 30)))[0]
            res = assign(c, net, compute_transfers=False)
            cp.append(res.C_p_all / sc[0])
            co.append(res.C_o / sc[1])
        cp_sd.append(np.std(cp))
        co_sd.append(np.std(co))
    a, b = np.mean(cp_sd), np.mean(co_sd)
    print(f"  sd putničkog člana C_p_all/donja_granica : {a:.3f}")
    print(f"  sd operaterskog člana C_o/MST           : {b:.3f}")
    print(f"  odnos uticaja: {a / b:.2f} : 1")
    print(f"  (sa starom skalom R*(max_len-1)*mean(tau) bilo je ~2.1 : 1)")


# --- 2. osetljivost na UNSERVED_FACTOR --------------------------------------
def check_unserved_factor(cs):
    hdr("2. osetljivost na UNSERVED_FACTOR (sada 20/5 = odnos brzina bus/pešak)")
    base = A.UNSERVED_FACTOR
    ratios = []
    for res, c in zip(greedy_results(cs[:8]), cs[:8]):
        m = np.isfinite(res.travel_time) & (c.street_shortest > 0)
        ratios.append(np.average(res.travel_time[m] / c.street_shortest[m],
                                 weights=c.demand[m]))
    print(f"  opslužen par putuje {np.mean(ratios):.2f}x duže od uličnog najkraćeg")
    print(f"  -> faktor mora biti osetno iznad toga, inače 'ne opslužiti' "
          f"košta koliko i 'opslužiti'\n")
    print(f"  {'faktor':>7} {'d_un':>7} {'C_p_all':>9} {'C_o':>7} {'cilj':>7}")
    try:
        for f in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
            A.UNSERVED_FACTOR = f
            d, cp, co, ob = [], [], [], []
            for c in cs[:8]:
                sc = cost_scales(c)
                res = assign(c, greedy_network(c, R, MIN_LEN, MAX_LEN,
                                               alpha=ALPHA)[0],
                             compute_transfers=False)
                d.append(res.d["d_un"])
                cp.append(res.C_p_all)
                co.append(res.C_o)
                ob.append(objective(res, sc, ALPHA))
            mark = "  <- u upotrebi" if f == base else ""
            print(f"  {f:>7.1f} {np.mean(d):>7.3f} {np.mean(cp):>9.2f} "
                  f"{np.mean(co):>7.1f} {np.mean(ob):>7.3f}{mark}")
    finally:
        A.UNSERVED_FACTOR = base
    print("  faktor bira tačku na osi pokrivenost/trošak; mora ići uz rezultate")


# --- 3. alpha stvarno pomera kompromis (postoji Pareto front) ---------------
def check_alpha_sweep(cs):
    hdr("3. pomera li alpha kompromis putnik/operater")
    print(f"  {'alpha':>6} {'d_un':>7} {'C_p_all':>9} {'C_o':>7}")
    for a in (0.1, 0.3, 0.5, 0.7, 0.9):
        rs = greedy_results(cs[:8], alpha=a)
        print(f"  {a:>6} {np.mean([r.d['d_un'] for r in rs]):>7.3f} "
              f"{np.mean([r.C_p_all for r in rs]):>9.2f} "
              f"{np.mean([r.C_o for r in rs]):>7.1f}")


# --- 4. invarijante koje su ranije mogle tiho da otkažu ---------------------
def check_invariants(cs):
    hdr("4. invarijante")
    for c in cs:
        assert c.validate() == [], (c.name, c.validate())
        assert np.isfinite(cost_scales(c)).all(), c.name
    print(f"  svi gradovi povezani, sve skale konačne ({len(cs)} gradova)")
    print("  (generator je ranije davao ~1% nepovezanih gradova, a nepovezan")
    print("   grad daje cp_scale=inf pa je stari cilj tiho gasio putnički član)")

    for res in greedy_results(cs[:6]):
        s = sum(res.d[k] for k in ("d_0", "d_1", "d_2", "d_3p", "d_un"))
        assert abs(s - 1) < 1e-9, s
    print("  udeli demanda po broju presedanja se sabiraju na 1")

    c = cs[0]
    nb = int(c.neighbors[0][0])
    assert any("duplirane" in p for p in
               TransitNetwork([[0, nb]] * R).check(c, R, MIN_LEN, MAX_LEN))
    print("  duplirane linije se odbijaju u check()")

    # baselines i RL sada optimizuju isti skalar
    from tndp.baselines.common import network_objective
    net = greedy_network(c, R, MIN_LEN, MAX_LEN, alpha=ALPHA)[0]
    sc = cost_scales(c)
    assert np.isclose(network_objective(c, net, sc, ALPHA),
                      objective(assign(c, net, compute_transfers=False), sc, ALPHA))
    print("  baseline cilj == RL cilj (isti skalar, ne leksikografski)")


# --- 5. uparena analiza umesto golih proseka -------------------------------
def check_paired(cs):
    hdr("5. uparena analiza (gradovi variraju više nego metode)")
    sc = [cost_scales(c) for c in cs]
    g = np.array([objective(assign(c, greedy_network(c, R, MIN_LEN, MAX_LEN,
                                                     alpha=ALPHA)[0],
                                   compute_transfers=False), s, ALPHA)
                  for c, s in zip(cs, sc)])
    h = np.array([objective(assign(c, hill_climb(c, R, MIN_LEN, MAX_LEN,
                                                 alpha=ALPHA)[0],
                                   compute_transfers=False), s, ALPHA)
                  for c, s in zip(cs, sc)])
    k = len(cs)
    d = g - h
    se_p = d.std(ddof=1) / np.sqrt(k)
    se_u = np.sqrt(g.var(ddof=1) / k + h.var(ddof=1) / k)
    print(f"  greedy       {g.mean():.3f} ± {g.std(ddof=1):.3f} (sd po gradovima)")
    print(f"  hill climbing {h.mean():.3f} ± {h.std(ddof=1):.3f}")
    print(f"  razlika {d.mean():+.3f}: nesparena greška ±{se_u:.3f}, "
          f"uparena ±{se_p:.3f} ({se_u / se_p:.1f}x uža)")


# --- 6. skala ulaza u mrežu i dekompozicija varijanse nagrade --------------
# feature tražnje mora imati ISTU raspodelu na treningu (gravity sintetika) i
# na testu (Mandl, Mumford), inače transfer trpi bez obzira na skalu
def check_input_scale():
    hdr("6. raspodela feature-a tražnje po instanci (mora biti ista svuda)")
    try:
        from scipy.stats import skew

        from tndp.rl.env import TndpEnv
        from tndp.rl.model import edge_tensors, node_features
    except ImportError:
        print("  torch nije instaliran, preskočeno")
        return
    from pathlib import Path
    data = Path(__file__).parent.parent / "data" / "benchmarks"
    sets = [("gravity (trening)", generate_city(seed=SEED_BASE, n_range=N_RANGE)),
            ("uniform", generate_city(seed=1, n_range=N_RANGE, demand_mode="uniform")),
            ("Mandl1", load_benchmark_city(data / "Mandl/Mandl1")),
            ("Mumford1", load_benchmark_city(data / "Mumford/Mumford1"))]
    print(f"  {'instanca':>18} {'node skew':>10} {'node max|z|':>12} "
          f"{'edge skew':>10} {'edge max|z|':>12}")
    for nm, c in sets:
        x = node_features(TndpEnv(c, R, MIN_LEN, MAX_LEN, ALPHA)).numpy()[:, 2]
        e = edge_tensors(c)[1].numpy()[:, 1]
        print(f"  {nm:>18} {skew(x):>10.2f} {np.abs(x).max():>12.2f} "
              f"{skew(e):>10.2f} {np.abs(e).max():>12.2f}")
    print("  (pre rang transformacije: gravity edge skew 5.35 vs Mumford 0.00)")


def check_reward_variance(cs):
    hdr("7. dekompozicija varijanse nagrade (koliko value(s0) uopšte može)")
    from tndp.baselines.common import random_route
    rows = []
    for c in cs[:8]:
        sc, rng, o = cost_scales(c), np.random.default_rng(0), []
        for _ in range(120):
            rs = []
            while len(rs) < R:
                r = random_route(c, rng, MIN_LEN, MAX_LEN)
                if r:
                    rs.append(r)
            o.append(-objective(assign(c, TransitNetwork(rs),
                                       compute_transfers=False), sc, ALPHA))
        rows.append(o)
    Rw = np.array(rows)
    within, between = Rw.var(axis=1).mean(), Rw.mean(axis=1).var()
    total = Rw.ravel().var()
    print(f"  ukupna varijansa nagrade  {total:.4f}")
    print(f"    unutar grada            {within:.4f}  ({within / total:.0%})  signal")
    print(f"    između gradova          {between:.4f}  ({between / total:.0%})  "
          f"gornja granica za value(s0)")
    print("  -> self-critical baseline (abl_selfcritical.yaml) hvata i deo")
    print("     unutar-gradske varijanse; verovatno bolji default od value glave")


def main():
    cs = cities()
    check_balance(cs)
    check_unserved_factor(cs)
    check_alpha_sweep(cs)
    check_invariants(cs)
    check_paired(cs)
    check_input_scale()
    check_reward_variance(cs)
    print("\nsve provere prošle")


if __name__ == "__main__":
    main()
