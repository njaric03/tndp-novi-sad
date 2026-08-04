# Ocena istih mreža sa frekvencijama. Trase su birane TRNDP ciljem (bez
# frekvencija), a ovde se nad njima radi druga faza: iz opterećenja po deonici
# se odrede intervali sleđenja, iz njih vreme čekanja putnika i broj vozila.
# Sve metode prolaze kroz isti postupak, pa je poređenje pošteno bez obzira
# što nijedna nije optimizovala flotu.
#
# pokretanje: python -m tndp.experiments.bench_freq runs/gravity-v1/best.pt

import argparse
import time
from pathlib import Path

import numpy as np

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core import frequencies as F
from tndp.core.assignment import assign, cost_scales, objective
from tndp.core.io import load_benchmark_city
from tndp.experiments.common import load_policy
from tndp.rl.evaluate import decode, decode_sampling

DATA = Path(__file__).parent.parent.parent / "data" / "benchmarks"

# (putanja, R, min_len, max_len, pretpostavljen dnevni broj putovanja).
# Poslednji broj je PRETPOSTAVKA, ne podatak: Mandl i Mumford objavljuju
# matricu tražnje bez perioda na koji se odnosi, a vršni sat se bez toga ne
# može odrediti. Vrednosti su red veličine grada te veličine; osetljivost je
# na dnu tabele i pokazuje da se poredak metoda ne menja.
INSTANCE = {
    "Mandl1": ("Mandl/Mandl1", 6, 2, 8, 30_000),
    "Mumford0": ("Mumford/Mumford0", 12, 2, 15, 100_000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--instances", nargs="+", default=list(INSTANCE), choices=list(INSTANCE))
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=32)
    args = ap.parse_args()

    policy, _ = load_policy(args.checkpoint)
    a = args.alpha

    lines = [
        f"# Frekvencije nad istim mrežama (model {args.checkpoint}, alpha={a})", "",
        "Trase su birane TRNDP ciljem (`cilj`, bez frekvencija). Druga faza dodeljuje",
        "intervale sleđenja iz opterećenja najopterećenije deonice, pa se dodela",
        "putnika ponavlja sa čekanjem od pola intervala umesto fiksnih 5 min.",
        "",
        f"Parametri: kapacitet {F.KAPACITET:.0f} putnika, udeo vršnog sata "
        f"{100 * F.UDEO_VRHA:.0f}%, interval u [{F.H_MIN:.0f}, {F.H_MAX:.0f}] min, "
        f"obrt {100 * F.OBRT:.0f}%.",
        "",
        "**Dnevni broj putovanja je pretpostavka.** Mandl i Mumford objavljuju matricu",
        "tražnje bez perioda na koji se odnosi, a bez perioda nema vršnog sata pa se",
        "intervali zalepe za donju ili gornju granicu. Matrica se zato preskalira na",
        "navedeni dnevni obim. Za Novi Sad to nije potrebno — tamo je period meren.",
        "",
        "`C_p` je bez čekanja (uporedivo sa ostalim tabelama), `C_p+čekanje` je isti",
        "prosek kad se čekanje uračuna, `flota` je broj vozila, `cilj_f` je",
        "alpha * (C_p_all sa čekanjem)/donja_granica + (1-alpha) * flota/donja_granica.",
        "",
        "| instanca | metoda | cilj | cilj_f | C_p | C_p+čekanje | čekanje | C_o | flota | med. interval | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for key in args.instances:
        rel, R, lo, hi, dnevno = INSTANCE[key]
        city = load_benchmark_city(DATA / rel)
        sc = cost_scales(city)

        metode = {
            "random 200": lambda c: random_search(c, R, lo, hi, num_samples=200, alpha=a)[0],
            "greedy": lambda c: greedy_network(c, R, lo, hi, alpha=a)[0],
            "hill climbing": lambda c: hill_climb(c, R, lo, hi, alpha=a)[0],
            "RL greedy dekod": lambda c: decode(policy, c, R, lo, hi, a)[0],
            f"RL sampling {args.samples}": lambda c: decode_sampling(
                policy, c, R, k=args.samples, min_len=lo, max_len=hi, alpha=a)[0],
        }

        print(f"\n== {key} (n={city.n}, R={R}, pretpostavljeno {dnevno:,} putovanja dnevno) ==")
        for naziv, resi in metode.items():
            t0 = time.perf_counter()
            net = resi(city)
            dt = time.perf_counter() - t0
            problems = net.check(city, R, lo, hi)
            if problems:
                print(f"  {naziv}: NEVALIDNO {problems[:2]}")
                continue
            base = assign(city, net)
            o = F.oceni(city, net, alpha=a, putovanja_dnevno=dnevno)
            row = (f"| {key} | {naziv} | {objective(base, sc, a):.3f} | {o['cilj']:.3f} | "
                   f"{base.C_p:.2f} | {o['res'].C_p:.2f} | {o['cekanje']:.2f} | "
                   f"{base.C_o:.0f} | {o['flota']:.0f} | {np.median(o['h']):.1f} | {dt:.1f} |")
            lines.append(row)
            print("  " + row)

    # osetljivost na dve konstante koje su izbor, ne podatak. greedy mreža je
    # dovoljna: pitanje je da li se zaključci menjaju, ne koja mreža je najbolja.
    lines += ["", "## Osetljivost na pretpostavke", "",
              "Greedy mreža na Mandl1, menja se po jedna konstanta.", "",
              "| konstanta | vrednost | flota | čekanje | med. interval |",
              "|---|---|---|---|---|"]
    rel, R, lo, hi, dnevno = INSTANCE["Mandl1"]
    city = load_benchmark_city(DATA / rel)
    net = greedy_network(city, R, lo, hi, alpha=a)[0]
    for kap in (60, 80, 120):
        o = F.oceni(city, net, alpha=a, kapacitet=kap, putovanja_dnevno=dnevno)
        lines.append(f"| kapacitet vozila | {kap} | {o['flota']:.0f} | "
                     f"{o['cekanje']:.2f} | {np.median(o['h']):.1f} |")
    for udeo in (0.08, 0.10, 0.12):
        o = F.oceni(city, net, alpha=a, udeo_vrha=udeo, putovanja_dnevno=dnevno)
        lines.append(f"| udeo vršnog sata | {100 * udeo:.0f}% | {o['flota']:.0f} | "
                     f"{o['cekanje']:.2f} | {np.median(o['h']):.1f} |")
    for pd in (dnevno // 2, dnevno, dnevno * 2):
        o = F.oceni(city, net, alpha=a, putovanja_dnevno=pd)
        lines.append(f"| putovanja dnevno | {pd:,} | {o['flota']:.0f} | "
                     f"{o['cekanje']:.2f} | {np.median(o['h']):.1f} |".replace(",", "."))

    out = Path(__file__).parent.parent.parent / "results" / "bench-freq.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsnimljeno u {out}")


if __name__ == "__main__":
    main()
