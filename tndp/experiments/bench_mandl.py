# random i greedy baselines na Mandlu; ispisuje markdown tabelu i snima je u results/bench-mandl.md pokretanje: python -m

import argparse

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign, cost_scales, objective
from tndp.core.io import load_benchmark_city, load_literature_solutions
from tndp.core.network import TransitNetwork
from tndp import BENCHMARKS, RESULTS

DATA = BENCHMARKS / "Mandl" / "Mandl1"

# standardna Mandl ogranicenja duzine linije (Mumford 2013, Holliday)
MIN_LEN, MAX_LEN = 2, 8


# C_p je ovde prosek nad opsluzenim parovima, jer je to konvencija u kojoj su objavljeni brojevi iz literature; sve
# `cilj` je skalar koji metode stvarno optimizuju, pa bez njega red tabele ne
# kaze da li je metoda losa ili samo bira drugu tacku fronta putnik/operater
def row(name, city, net, scales, alpha):
    res = assign(city, net)
    d = res.d
    return (
        f"| {name} | {objective(res, scales, alpha):.3f} | {res.C_p:.2f} | {res.C_o:.0f} | "
        f"{d['d_0']:.2f} | {d['d_1']:.2f} | {d['d_2']:.2f} | {d['d_un']:.2f} |"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--num-routes", type=int, nargs="+", default=[4, 6, 7, 8])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    city = load_benchmark_city(DATA)
    scales = cost_scales(city)
    solutions = load_literature_solutions(
        DATA / "literature_solutions_for_mandl1_20181025.txt"
    )

    def red(name, net):
        return row(name, city, net, scales, args.alpha)

    lines = [
        f"# Mandl baselines (alpha={args.alpha}, {args.samples} random uzoraka, seed {args.seed})",
        "",
        "Ova tabela služi kao provera implementacije assignment-a naspram objavljenih",
        "vrednosti, ne kao poređenje metoda: red iz literature je optimizovan po svom",
        "cilju, a ne po `cilj` koloni odavde.",
        "",
        "`cilj` je skalar koji sve metode iz repoa stvarno minimizuju. `C_p` je prosek",
        "samo nad opsluženim parovima, konvencija u kojoj su objavljeni brojevi iz",
        "literature. Lokalna pretraga zato ume da ima lošiji `C_p` a bolji `cilj` od",
        "konstruktivne heuristike: kupuje kraće linije (`C_o`) po ceni dužih putovanja.",
        "",
        "| metoda | cilj | C_p (min) | C_o (min) | d_0 | d_1 | d_2 | d_un |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in args.num_routes:
        rand_net, _ = random_search(
            city, r, MIN_LEN, MAX_LEN,
            num_samples=args.samples, alpha=args.alpha, seed=args.seed,
        )
        greedy_net, _ = greedy_network(city, r, MIN_LEN, MAX_LEN, alpha=args.alpha)
        climb_net, _ = hill_climb(city, r, MIN_LEN, MAX_LEN, alpha=args.alpha,
                                  seed=args.seed)
        lines.append(red(f"random najbolja od {args.samples}, R={r}", rand_net))
        lines.append(red(f"greedy, R={r}", greedy_net))
        lines.append(red(f"hill climbing, R={r}", climb_net))
        ref = f"Nikolic (2013) {r} routes"
        if ref in solutions:
            lines.append(red(f"literatura: {ref}", TransitNetwork(solutions[ref])))
    for name in ["Mumford (2013) 6 best passenger", "Mumford (2013) 6 best operator"]:
        lines.append(red(f"literatura: {name}", TransitNetwork(solutions[name])))

    table = "\n".join(lines)
    print(table)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bench-mandl.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
