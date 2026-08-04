# random i greedy baselines na Mandlu; ispisuje markdown tabelu
# i snima je u results/bench-mandl.md
# pokretanje: python -m tndp.experiments.bench_mandl [--alpha 0.5] [--samples 1000]

import argparse
from pathlib import Path

from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign
from tndp.core.io import load_benchmark_city, load_literature_solutions
from tndp.core.network import TransitNetwork

DATA = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "Mandl" / "Mandl1"

# standardna Mandl ogranicenja duzine linije (Mumford 2013, Holliday)
MIN_LEN, MAX_LEN = 2, 8


# C_p je ovde prosek nad opsluzenim parovima, jer je to konvencija u kojoj
# su objavljeni brojevi iz literature; sve objavljene mreze su povezane
# (d_un = 0) pa je C_p == C_p_all i poredjenje je korektno
def row(name: str, city, net: TransitNetwork) -> str:
    res = assign(city, net)
    d = res.d
    return (
        f"| {name} | {res.C_p:.2f} | {res.C_o:.0f} | "
        f"{d['d_0']:.2f} | {d['d_1']:.2f} | {d['d_2']:.2f} | {d['d_un']:.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--num-routes", type=int, nargs="+", default=[4, 6, 7, 8])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    city = load_benchmark_city(DATA)
    solutions = load_literature_solutions(
        DATA / "literature_solutions_for_mandl1_20181025.txt"
    )

    lines = [
        f"# Mandl baselines (alpha={args.alpha}, {args.samples} random uzoraka, seed {args.seed})",
        "",
        "| metoda | C_p (min) | C_o (min) | d_0 | d_1 | d_2 | d_un |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in args.num_routes:
        rand_net, _ = random_search(
            city, r, MIN_LEN, MAX_LEN,
            num_samples=args.samples, alpha=args.alpha, seed=args.seed,
        )
        greedy_net, _ = greedy_network(city, r, MIN_LEN, MAX_LEN, alpha=args.alpha)
        climb_net, _ = hill_climb(city, r, MIN_LEN, MAX_LEN, alpha=args.alpha,
                                  seed=args.seed)
        lines.append(row(f"random najbolja od {args.samples}, R={r}", city, rand_net))
        lines.append(row(f"greedy, R={r}", city, greedy_net))
        lines.append(row(f"hill climbing, R={r}", city, climb_net))
        ref = f"Nikolic (2013) {r} routes"
        if ref in solutions:
            lines.append(row(f"literatura: {ref}", city, TransitNetwork(solutions[ref])))
    for name in ["Mumford (2013) 6 best passenger", "Mumford (2013) 6 best operator"]:
        lines.append(row(f"literatura: {name}", city, TransitNetwork(solutions[name])))

    table = "\n".join(lines)
    print(table)
    out = Path(__file__).parent.parent.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "bench-mandl.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
