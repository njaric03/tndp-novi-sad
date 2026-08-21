# Glavni rezultat studije slucaja: politika trenirana na SINTETICI pusta se na graf Novog Sada u jednom prolazu

import argparse
import time
from pathlib import Path


from tndp.baselines.greedy import greedy_network
from tndp.baselines.hill_climb import hill_climb
from tndp.baselines.random_search import random_search
from tndp.core.assignment import assign, cost_scales, objective
from tndp.experiments.common import load_policy
from tndp.novisad.instanca import gsp_mreza, ucitaj
from tndp.rl.evaluate import decode, decode_sampling

KOREN = Path(__file__).resolve().parent.parent.parent
REZULTATI = KOREN / "results"
# isti run iz kog su i sve tabele o Novom Sadu u radu; novisad-r19 je stariji,
# na stopi ucenja 1e-4, i njegovi brojevi se ne smeju naci na istoj strani
MODEL = KOREN / "runs" / "novisad-r19h" / "best.pt"


# koliko se mreza poklapa sa GSP-om, mereno na neusmerenim parovima uzastopnih zona
def _preklapanje(mreza, gsp):
    def ivice(net):
        return {frozenset((a, b)) for r in net.routes for a, b in zip(r, r[1:])}

    a, b = ivice(mreza), ivice(gsp)
    return {"deljenih ivica": len(a & b), "ivica metode": len(a),
            "Jaccard": len(a & b) / len(a | b) if a | b else 0.0}


def _red(ime, net, city, scales, alpha, dt, gsp):
    res = assign(city, net)
    p = _preklapanje(net, gsp)
    return {"metoda": ime, "cilj": objective(res, scales, alpha),
            "C_p_all": res.C_p_all, "C_p": res.C_p, "C_o": res.C_o,
            "d_0": res.d["d_0"], "d_un": res.d["d_un"],
            "deljenih": p["deljenih ivica"], "ivica": p["ivica metode"],
            "jaccard": p["Jaccard"], "s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=str(MODEL))
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--samples", type=int, default=32)
    args = ap.parse_args()

    city, imena = ucitaj()
    city.require_valid()
    gsp, dnevnik = gsp_mreza(city, imena)
    R = len(gsp.routes)
    duz = [len(r) for r in gsp.routes]
    lo, hi = min(duz), max(duz)

    policy, cfg = load_policy(args.checkpoint)
    a = args.alpha if args.alpha is not None else cfg["alpha_eval"]
    scales = cost_scales(city)

    print(f"Novi Sad: n={city.n}, R={R}, dužina linije [{lo}, {hi}]")
    print(f"model treniran na n {cfg['n_range']}, R={cfg['num_routes']}, "
          f"dužina [{cfg['min_len']}, {cfg['max_len']}], alpha={a}")

    metode = {
        "GSP (postojeća)": lambda: gsp,
        "random 200": lambda: random_search(city, R, lo, hi, num_samples=200, alpha=a)[0],
        "greedy": lambda: greedy_network(city, R, lo, hi, alpha=a)[0],
        "hill climbing": lambda: hill_climb(city, R, lo, hi, alpha=a)[0],
        "RL greedy dekod": lambda: decode(policy, city, R, lo, hi, a)[0],
        f"RL sampling {args.samples}": lambda: decode_sampling(
            policy, city, R, k=args.samples, min_len=lo, max_len=hi, alpha=a)[0],
    }

    redovi = []
    for ime, resi in metode.items():
        t0 = time.perf_counter()
        net = resi()
        dt = time.perf_counter() - t0
        prekrsaji = net.check(city, R, lo, hi)
        if prekrsaji:
            print(f"  {ime}: NEVALIDNO {prekrsaji[:2]}")
            continue
        r = _red(ime, net, city, scales, a, dt, gsp)
        redovi.append(r)
        print(f"  {ime:20s} cilj {r['cilj']:.3f}  C_p {r['C_p']:5.2f}  "
              f"C_o {r['C_o']:6.1f}  d_un {r['d_un']:.3f}  "
              f"Jaccard {r['jaccard']:.3f}  ({dt:.1f} s)")

    _izvestaj(redovi, city, R, lo, hi, cfg, a, args.checkpoint)


def _izvestaj(redovi, city, R, lo, hi, cfg, a, ckpt):
    gsp_cilj = next(r["cilj"] for r in redovi if r["metoda"].startswith("GSP"))
    r = ["# Novi Sad: model naspram postojeće GSP mreže", "",
         f"Politika je trenirana na SINTETIČKIM gradovima (n {cfg['n_range']}, "
         f"R={cfg['num_routes']}, dužina linije [{cfg['min_len']}, {cfg['max_len']}])",
         "i puštena na Novi Sad u jednom prolazu, bez dotreniravanja. To je tvrdnja",
         "koju rad proverava.", "",
         f"Instanca: n={city.n} zona, R={R} linija, dužina [{lo}, {hi}], alpha={a}.",
         f"Model: `{ckpt}`.", "",
         "`cilj` je isti skalar za sve metode, manje je bolje. `Jaccard` meri",
         "poklapanje sa GSP mrežom po parovima uzastopnih zona, koliko model",
         "gradi iste koridore kao stvarni planeri.", "",
         "| metoda | cilj | vs GSP | C_p_all | C_p | C_o | d_0 | d_un | Jaccard | s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for x in redovi:
        vs = "-" if x["metoda"].startswith("GSP") else f"{gsp_cilj - x['cilj']:+.3f}"
        r.append(f"| {x['metoda']} | {x['cilj']:.3f} | {vs} | {x['C_p_all']:.2f} | "
                 f"{x['C_p']:.2f} | {x['C_o']:.1f} | {x['d_0']:.2f} | "
                 f"{x['d_un']:.3f} | {x['jaccard']:.3f} | {x['s']:.1f} |")
    r += ["", "`vs GSP` je razlika u cilju prema postojećoj mreži; pozitivno znači",
          "bolje od GSP-a po ovom cilju. Treba ga čitati uz ograničenja niže.", "",
          "## Šta ovo poređenje NE kaže", "",
          "GSP mreža nije projektovana po ovom cilju, pa je porediti po njemu je",
          "delom nepravedno u oba smera:", "",
          "- cilj ne vidi frekvencije, kapacitet ni vozni park, a GSP linije postoje",
          "  u režimu u kom te stvari odlučuju (vidi results/novisad-frekvencije.md)",
          "- GSP trase su prevedene u zonski graf, pri čemu su svedene na proste",
          "  puteve i popunjene po susedstvu (results koje daje tndp/novisad/instanca.py);",
          "  to je aproksimacija stvarne trase",
          "- mreža koja postoji nosi i ograničenja koja model ne zna: infrastrukturu,",
          "  okretnice, kolektivne ugovore, istorijske odluke", "",
          "Zato je `Jaccard` uz `cilj` bitniji nego sam `cilj`: on kaže da li model",
          "prepoznaje iste koridore, što je tvrdnja koja ne zavisi od toga da li je",
          "naša funkcija cilja ista kao ona koju je GSP imao na umu.", ""]
    REZULTATI.mkdir(exist_ok=True)
    (REZULTATI / "novisad-poredjenje.md").write_text("\n".join(r) + "\n", encoding="utf-8")
    print(f"\n-> {REZULTATI / 'novisad-poredjenje.md'}")


if __name__ == "__main__":
    main()
