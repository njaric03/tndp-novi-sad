# Uparen test izmedju MODELA (ne model vs greedy kao u ablacijama) - da li
# varijanta stvarno odstupa od osnovne politike. Wilcoxon + Holm za vise varijanti.
#
# pokretanje:
#   python -m tndp.experiments.compare_models runs/sweep-lr1e3/best.pt \
#       runs/sweep-lr1e3-s1/best.pt runs/abl-akm-h/best.pt --cities 20

import argparse
import time
from pathlib import Path

import numpy as np

from tndp.core.assignment import assign, cost_scales, objective
from tndp.experiments.common import (fmt_p, held_out_cities, holm, load_policy,
                                     paired_vs, write_table)
from tndp.rl.evaluate import decode_sampling


# vrednost cilja po gradu, da se moze uparivati
def per_city(checkpoint, cities, k, alpha):
    policy, cfg = load_policy(checkpoint)
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    v, t0 = [], time.perf_counter()
    for c in cities:
        net, _ = decode_sampling(policy, c, R, k=k, min_len=lo, max_len=hi, alpha=alpha)
        v.append(objective(assign(c, net), cost_scales(c), alpha))
    return np.array(v), (time.perf_counter() - t0) / len(cities), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--cities", type=int, default=20)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default="uporedjenje-modela")
    args = ap.parse_args()

    # gradovi se prave iz configa REFERENTNOG modela, da svi vide isti skup
    _, cfg0 = load_policy(args.reference)
    cities = held_out_cities(cfg0, args.cities)
    ref_v, ref_s, _ = per_city(args.reference, cities, args.samples, args.alpha)
    print(f"{args.cities} gradova, n {cfg0['n_range']}, R={cfg0['num_routes']}, "
          f"alpha={args.alpha}, sampling {args.samples}")
    print(f"referenca {args.reference}: cilj {ref_v.mean():.3f} +- {ref_v.std(ddof=1):.3f}")

    rows = []
    for ck in args.variants:
        v, s, cfg = per_city(ck, cities, args.samples, args.alpha)
        d, se, p = paired_vs(v, ref_v)
        rows.append({"model": ck, "cilj": v.mean(), "sd": v.std(ddof=1),
                     "delta": d, "se": se, "p": p, "s": s,
                     "featuri": cfg.get("features", "v1")})
    for r, pa in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = pa
        print(f"  {Path(r['model']).parts[-2]:22s} cilj {r['cilj']:.3f}  "
              f"delta {r['delta']:+.3f} +- {r['se']:.3f}  p {fmt_p(r['p'])}  "
              f"Holm {fmt_p(pa)}")

    _report(args, cfg0, ref_v, rows)


def _report(args, cfg0, ref_v, rows):
    r = ["# Uparen test između modela", "",
         "Svaki model je pušten na ISTIH " + str(args.cities) + " held-out gradova, pa je",
         "razlika u `cilj` testirana uparenim Wilcoxonom protiv referentnog MODELA, ne",
         "protiv greedyja. Ablacione tabele porede sa greedyjem, što ne odgovara na",
         "pitanje da li se varijanta razlikuje od osnovne politike.", "",
         "`Holm` je p vrednost korigovana za broj varijanti u tabeli. Bez korekcije je",
         f"pri {len(rows)} testova i pragu 0.05 jedan lažno pozitivan očekivan.", "",
         f"Gradovi: n {cfg0['n_range']}, R={cfg0['num_routes']}, alpha={args.alpha}, "
         f"sampling {args.samples}.", "",
         f"Referenca `{args.reference}`: cilj {ref_v.mean():.3f} ± {ref_v.std(ddof=1):.3f}.", "",
         "| model | featuri | cilj | Δ vs referenca | p | p (Holm) | s/grad |",
         "|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda z: z["cilj"]):
        r.append(f"| `{Path(x['model']).parts[-2]}` | {x['featuri']} | "
                 f"{x['cilj']:.3f} ± {x['sd']:.3f} | {x['delta']:+.3f} ± {x['se']:.3f} | "
                 f"{fmt_p(x['p'])} | {fmt_p(x['p_holm'])} | {x['s']:.1f} |")
    r += ["", "Δ>0 znači da je varijanta bolja od reference.", "",
          "## Kako se ovo čita", "",
          "Uparen test je osetljiviji od poređenja proseka, ali odgovara na uže",
          "pitanje: da li su se DVA TRENINGA razlikovala. Ako je u tabeli i replika",
          "iste konfiguracije sa drugim seedom, ona služi kao kontrola. Značajna",
          "razlika kod nje znači da p vrednost meri seed, ne izmenu koja se ispituje,",
          "i da nijedan red u tabeli ne sme da se čita kao efekat featura.", "",
          "Za tvrdnju o featuru treba više seedova po varijanti, pa poređenje",
          "raspodela umesto pojedinačnih runova.", ""]
    write_table(f"{args.out}.md", r)


if __name__ == "__main__":
    main()
