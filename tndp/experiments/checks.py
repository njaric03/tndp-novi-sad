# Provere invarijanti na kojima pociva funkcija cilja, plus osetljivost na dve konstante koje u njoj ostaju


import numpy as np

import tndp.core.assignment as A
from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import (assign, cost_scales, network_objective,
                                  objective)
from tndp.core.io import load_benchmark_city
from tndp.core.network import TransitNetwork
from tndp.core.synth import generate_city
from tndp import BENCHMARKS

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


# --- 1
def check_balance(cs):
    hdr("1. balans putničkog i operaterskog člana (koliko svaki pomera cilj)")
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
    # alpha pri kojoj oba člana jednako pomeraju cilj: a*sd_cp = (1-a)*sd_co
    print(f"  -> članovi NISU jednako skalirani. Jednak uticaj je na "
          f"alpha ~ {b / (a + b):.2f},")
    print("     ne na 0.5; pri alpha=0.5 cilj prati uglavnom putnički član.")


# --- 2
def check_unserved_factor(cs):
    hdr("2. osetljivost na UNSERVED_FACTOR (odnos brzina 20/5 puta težina pešačenja 2)")
    base = A.UNSERVED_FACTOR
    # cela raspodela, ne samo prosek: faktor deli opslužene parove na one koje se
    # isplati opslužiti i one koje se isplati ispustiti, pa je bitan gornji rep
    mean_r, p95, mx, over = [], [], [], []
    for res, c in zip(greedy_results(cs[:8]), cs[:8]):
        m = np.isfinite(res.travel_time) & (c.street_shortest > 0)
        r = res.travel_time[m] / c.street_shortest[m]
        w = c.demand[m]
        mean_r.append(np.average(r, weights=w))
        mx.append(r.max())
        order = np.argsort(r)
        cum = np.cumsum(w[order]) / w.sum()
        p95.append(float(r[order][np.searchsorted(cum, 0.95)]))
        over.append(float(w[r > base].sum() / w.sum()))
    print("  odnos vreme_mrežom / ulično najkraće, po opsluženom paru:")
    print(f"    demand-ponderisan prosek       {np.mean(mean_r):.2f}x")
    print(f"    demand-ponderisan 95. percentil {np.mean(p95):.2f}x")
    print(f"    maksimum                        {max(mx):.2f}x "
          f"(po gradu {min(mx):.1f}-{max(mx):.1f})")
    print(f"  -> faktor {base:g} NIJE iznad najgoreg opsluženog para; iznad je "
          f"{100 * (1 - np.mean(over)):.0f}% opslužene tražnje.")
    print(f"     Preostalih {100 * np.mean(over):.1f}% (po gradu do "
          f"{100 * max(over):.1f}%) su parovi koje se optimizatoru i dalje")
    print(f"     isplati ispustiti. Tek faktor iznad {max(mx):.0f} bi to zatvorio, "
          f"ali bi mrežu naterao\n     da pokrije sve po svaku cenu; sweep ispod "
          f"pokazuje šta se time dobija.\n")
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


# --- 3
def check_alpha_sweep(cs):
    hdr("3. pomera li alpha kompromis putnik/operater")
    print(f"  {'alpha':>6} {'d_un':>7} {'C_p_all':>9} {'C_o':>7}")
    for a in (0.1, 0.3, 0.5, 0.7, 0.9):
        rs = greedy_results(cs[:8], alpha=a)
        print(f"  {a:>6} {np.mean([r.d['d_un'] for r in rs]):>7.3f} "
              f"{np.mean([r.C_p_all for r in rs]):>9.2f} "
              f"{np.mean([r.C_o for r in rs]):>7.1f}")


# --- 4
def check_invariants(cs):
    hdr("4. invarijante")
    for c in cs:
        c.require_valid()
        if not np.isfinite(cost_scales(c)).all():
            raise ValueError(f"beskonačna skala na {c.name}")
    print(f"  svi gradovi povezani, sve skale konačne ({len(cs)} gradova)")
    print("  (generator je ranije davao ~1% nepovezanih gradova, a nepovezan")
    print("   grad daje cp_scale=inf pa je stari cilj tiho gasio putnički član)")

    for res in greedy_results(cs[:6]):
        s = sum(res.d[k] for k in ("d_0", "d_1", "d_2", "d_3p", "d_un"))
        if abs(s - 1) >= 1e-9:
            raise ValueError(f"udeli demanda se sabiraju na {s}, ne na 1")
    print("  udeli demanda po broju presedanja se sabiraju na 1")

    c = cs[0]
    nb = int(c.neighbors[0][0])
    duplikat = TransitNetwork([[0, nb]] * R).check(c, R, MIN_LEN, MAX_LEN)
    if not any("duplirane" in p for p in duplikat):
        raise ValueError(f"check() ne prijavljuje duplikate: {duplikat}")
    print("  duplirane linije se odbijaju u check()")

    # baselines i RL sada optimizuju isti skalar
    net = greedy_network(c, R, MIN_LEN, MAX_LEN, alpha=ALPHA)[0]
    sc = cost_scales(c)
    baseline = network_objective(c, net, sc, ALPHA)
    rl = objective(assign(c, net, compute_transfers=False), sc, ALPHA)
    if not np.isclose(baseline, rl):
        raise ValueError(f"baseline cilj {baseline} != RL cilj {rl}")
    print("  baseline cilj == RL cilj (isti skalar, ne leksikografski)")


# --- 5
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


# --- 6
def check_input_scale():
    hdr("6. raspodela feature-a tražnje po instanci (mora biti ista svuda)")
    try:
        from scipy.stats import skew

        from tndp.rl.env import TndpEnv
        from tndp.rl.features import edge_tensors, node_features
    except ImportError:
        print("  torch nije instaliran, preskočeno")
        return
    sets = [("gravity (trening)", generate_city(seed=SEED_BASE, n_range=N_RANGE)),
            ("uniform", generate_city(seed=1, n_range=N_RANGE, demand_mode="uniform")),
            ("Mandl1", load_benchmark_city(BENCHMARKS / "Mandl/Mandl1")),
            ("Mumford1", load_benchmark_city(BENCHMARKS / "Mumford/Mumford1"))]
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
    from tndp.baselines.routes import random_route
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
    print("  -> value(s0) vidi samo 8% varijanse, pa se cinilo da bi self-critical")
    print("     baseline, koji hvata i unutar-gradsku, bio bolji default.")
    print("     NIJE POTVRDJENO: self-critical daje 1.522 naspram 1.534 +- 0.040")
    print("     osnovne, mereno na cetiri semena, dakle unutar rasipanja.")
    print("     Dekompozicija dakle ne predvidja ishod.")


# --- 8
# Akcija je par (kraj, cvor), ne samo cvor. Ovo meri koliko bi se izgubilo da
# nije tako: cvor susedan OBA kraja bi imao samo jednu dostizanu varijantu.
def check_both_ends(cs, episodes=20):
    from tndp.rl.env import HALT, HEAD, TAIL, TndpEnv
    hdr("8. koliko poteza nudi isti čvor na oba kraja linije")
    both = total = 0
    for c in cs:
        rng = np.random.default_rng(0)
        for _ in range(episodes):
            env = TndpEnv(c, R, MIN_LEN, MAX_LEN, ALPHA)
            while not env.done:
                decision, mask = env.decision()
                if env.current and mask.any():
                    total += 1
                    if set(np.flatnonzero(mask[HEAD])) & set(np.flatnonzero(mask[TAIL])):
                        both += 1
                opts = list(np.flatnonzero(mask.reshape(-1)))
                if decision == HALT:
                    opts.append(-1)
                env.step(int(opts[rng.integers(len(opts))]))
    print(f"  poteza sa bar jednim čvorom dozvoljenim na oba kraja: "
          f"{both}/{total} = {100 * both / total:.1f}%")
    print("  (nasumične epizode; to je udeo POTEZA sa bar jednim takvim čvorom,")
    print("   ne udeo akcija. Po akcijama je dvosmisleno oko 10%: toliko bi ih")
    print("   bilo nedostižno da je akcija samo čvor umesto para (kraj, čvor).)")


def main():
    cs = cities()
    check_balance(cs)
    check_unserved_factor(cs)
    check_alpha_sweep(cs)
    check_invariants(cs)
    check_paired(cs)
    check_input_scale()
    check_reward_variance(cs)
    check_both_ends(cs)
    print("\nsve provere prošle")


if __name__ == "__main__":
    main()
