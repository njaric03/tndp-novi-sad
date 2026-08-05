# Uparen test izmedju MODELA, na istim gradovima.
#
# Ablacione tabele porede svaku varijantu sa greedyjem, sto ne odgovara na
# pitanje zbog kog ablacija i postoji: da li se varijanta razlikuje od OSNOVNE
# POLITIKE. Ovde se svaki model pusta na isti skup held-out gradova i poredi
# uparenim Wilcoxonom sa referentnim modelom, uz Holm korekciju jer se testira
# vise varijanti odjednom.
#
# pokretanje:
#   python -m tndp.experiments.uporedi_modele runs/gravity-v1/best.pt \
#       runs/abl-betweenness/best.pt runs/abl-coreness/best.pt --cities 20

import argparse
import time
from pathlib import Path

import numpy as np

from tndp.core.assignment import assign, cost_scales, objective
from tndp.experiments.common import (SEED_BASE, fmt_p, held_out_cities, holm,
                                     load_policy, paired_vs)
from tndp.rl.evaluate import decode_sampling

KOREN = Path(__file__).resolve().parent.parent.parent
REZULTATI = KOREN / "results"


# vrednost cilja po gradu, da se moze uparivati
def po_gradu(checkpoint, cities, k, alpha):
    policy, cfg = load_policy(checkpoint)
    R, lo, hi = cfg["num_routes"], cfg["min_len"], cfg["max_len"]
    v, t0 = [], time.perf_counter()
    for c in cities:
        net, _ = decode_sampling(policy, c, R, k=k, min_len=lo, max_len=hi, alpha=alpha)
        v.append(objective(assign(c, net), cost_scales(c), alpha))
    return np.array(v), (time.perf_counter() - t0) / len(cities), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("referenca")
    ap.add_argument("varijante", nargs="+")
    ap.add_argument("--cities", type=int, default=20)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default="uporedjenje-modela")
    args = ap.parse_args()

    # gradovi se prave iz configa REFERENTNOG modela, da svi vide isti skup
    _, cfg0 = load_policy(args.referenca)
    cities = held_out_cities(cfg0, args.cities)
    ref_v, ref_s, _ = po_gradu(args.referenca, cities, args.samples, args.alpha)
    print(f"{args.cities} gradova, n {cfg0['n_range']}, R={cfg0['num_routes']}, "
          f"alpha={args.alpha}, sampling {args.samples}")
    print(f"referenca {args.referenca}: cilj {ref_v.mean():.3f} +- {ref_v.std(ddof=1):.3f}")

    redovi = []
    for ck in args.varijante:
        v, s, cfg = po_gradu(ck, cities, args.samples, args.alpha)
        d, se, p = paired_vs(v, ref_v)
        redovi.append({"model": ck, "cilj": v.mean(), "sd": v.std(ddof=1),
                       "delta": d, "se": se, "p": p, "s": s,
                       "featuri": cfg.get("features", "v1")})
    for r, pa in zip(redovi, holm([r["p"] for r in redovi])):
        r["p_holm"] = pa
        print(f"  {Path(r['model']).parts[-2]:22s} cilj {r['cilj']:.3f}  "
              f"delta {r['delta']:+.3f} +- {r['se']:.3f}  p {fmt_p(r['p'])}  "
              f"Holm {fmt_p(pa)}")

    _izvestaj(args, cfg0, ref_v, redovi)


def _izvestaj(args, cfg0, ref_v, redovi):
    r = ["# Uparen test izmedju modela", "",
         "Svaki model je pusten na ISTIH " + str(args.cities) + " held-out gradova, pa je",
         "razlika u `cilj` testirana uparenim Wilcoxonom protiv referentnog MODELA, ne",
         "protiv greedyja. Ablacione tabele porede sa greedyjem, sto ne odgovara na",
         "pitanje da li se varijanta razlikuje od osnovne politike.", "",
         "`Holm` je p vrednost korigovana za broj varijanti u tabeli. Bez korekcije je",
         f"pri {len(redovi)} testova i pragu 0.05 jedan lazno pozitivan ocekivan.", "",
         f"Gradovi: n {cfg0['n_range']}, R={cfg0['num_routes']}, alpha={args.alpha}, "
         f"sampling {args.samples}.", "",
         f"Referenca `{args.referenca}`: cilj {ref_v.mean():.3f} ± {ref_v.std(ddof=1):.3f}.", "",
         "| model | featuri | cilj | Δ vs referenca | p | p (Holm) | s/grad |",
         "|---|---|---|---|---|---|---|"]
    for x in sorted(redovi, key=lambda z: z["cilj"]):
        r.append(f"| `{Path(x['model']).parts[-2]}` | {x['featuri']} | "
                 f"{x['cilj']:.3f} ± {x['sd']:.3f} | {x['delta']:+.3f} ± {x['se']:.3f} | "
                 f"{fmt_p(x['p'])} | {fmt_p(x['p_holm'])} | {x['s']:.1f} |")
    r += ["", "Δ>0 znaci da je varijanta bolja od reference.", "",
          "## Kako se ovo cita", "",
          "Uparen test je osetljiviji od poredjenja proseka, ali odgovara na uze",
          "pitanje: da li su se DVA TRENINGA razlikovala. Ako je u tabeli i replika",
          "iste konfiguracije sa drugim seedom, ona sluzi kao kontrola. Znacajna",
          "razlika kod nje znaci da p vrednost meri seed, ne izmenu koja se ispituje,",
          "i da nijedan red u tabeli ne sme da se cita kao efekat featura.", "",
          "Za tvrdnju o featuru treba vise seedova po varijanti, pa poredjenje",
          "raspodela umesto pojedinacnih runova.", ""]
    REZULTATI.mkdir(exist_ok=True)
    (REZULTATI / f"{args.out}.md").write_text("\n".join(r) + "\n", encoding="utf-8")
    print(f"\n-> {REZULTATI / (args.out + '.md')}")


if __name__ == "__main__":
    main()
