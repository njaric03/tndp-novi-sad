# Dizajn mreže linija javnog prevoza pomoću GNN + RL

Dat je graf grada sa uličnom mrežom i procenom koliko se putuje između zona, a traži se
skup autobuskih linija koji putnike vozi brzo a prevoznika ne košta previše. To je
Transit Network Design Problem, NP-težak je, i obično se rešava metaheuristikom koja za
svaki grad pretražuje iznova. Ovde umesto pretrage stoji graf neuronska mreža: nauči
heuristiku na sintetičkim gradovima, pa novom gradu da mrežu u jednom prolazu.

Politika je GATv2 sa dinamičkom pažnjom, trenirana REINFORCE-om, a meri se protiv
nasumične pretrage, konstruktivne heuristike i lokalne pretrage: na sintetici koju nije
videla, na Mandlu i Mumfordovim instancama, i na Novom Sadu sklopljenom iz otvorenih
podataka.

## Funkcija cilja

[assignment.py](tndp/core/assignment.py) pusti putnike kroz graf linija i za svaki par
zona nađe najbrži put, uz penal od 5 min po presedanju. Iz toga izlazi:

- `C_p_all`, prosečno vreme nad svim parovima, gde nepokriven par plaća `UNSERVED_FACTOR = 8` puta ulično najkraće vreme.
- `C_p`, isti prosek ali samo nad opsluženim parovima.
- `C_o`, ukupno vreme vožnje svih linija u jednom smeru.
- `d_0/d_1/d_2/d_un`, udeli tražnje bez presedanja, sa jednim, sa dva, i nepokrivene.

Cilj je

```
alpha * C_p_all / (donja granica C_p)  +  (1 - alpha) * C_o / (vreme MST-a)
```

## Pokretanje

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev,rl]     # dodatno .[geo] za Novi Sad, .[nb] za notebooks
pytest
python -m tndp.experiments.checks   # invarijante i osetljivost na konstante
```

Tri testa u `tests/`: assignment, frekvencije, i vrednost cilja naspram objavljenih
Mandl vrednosti.

### Trening

```bash
python -m tndp.rl.train --config configs/gravity-v2h.yaml          # runs/<ime>/best.pt
python -m tndp.rl.train --config configs/gravity-v1.yaml --seed 1
```

Evaluacija koristi `best.pt`, najbolji na validaciji; `policy.pt` je poslednji.

### Novi Sad

Moduli u `tndp/novisad/` se pokreću sa `python -m tndp.novisad.<ime>`, i to ovim redom:

```
preuzmi → sredi → zone → sadrzaji → ulice → traznja
pa onda kalibracija, frekvencije, poredjenje, karta
```

Redosled je obavezan, jer svaki korak čita ono što je prethodni upisao. `data/novisad/`
nije u repou nego se ovim lancem i pravi.

## Rezultati

Glavni modeli su `runs/gravity-v2h` (sintetika, benchmark instance) i `runs/novisad-r19h`
(Novi Sad), oba na stopi učenja `1e-3`.

Sadržaj `results/` nije u repou, prave ga skripte iz `tndp/experiments/`. Uzimaju
putanju do checkpointa, a zastavice su u `--help`.

## Struktura

```
tndp/
  core/        graf grada, mreža linija, assignment, cilj, frekvencije, synth.py
  baselines/   nasumična pretraga, konstruktivna heuristika, lokalna pretraga
  rl/          MDP env, GATv2 model, REINFORCE, dekoderi, MCTS
  novisad/     podaci, zoniranje, tražnja, kalibracija, karte
  experiments/ skripte koje pokreću metode i pišu results/
  viz/         crtanje
configs/       yaml konfiguracije treninga i ablacija
data/benchmarks/  Mandl i Mumford instance
notebooks/     tri sveske nad postojećim kodom: podaci, model, rezultati
tests/         assignment, frekvencije, acceptance na Mandlu
```

## Izvori

Metoda je rađena po radu Hollidaya, El-Geneidyja i Dudeka, [Learning Heuristics for
Transit Network Design and Improvement with Deep Reinforcement
Learning](https://arxiv.org/abs/2404.05894). Instance u `data/benchmarks/` su Mandl i
Mumford, preuzete iz
[RenatoArbex/TransitNetworkDesign](https://github.com/RenatoArbex/TransitNetworkDesign).
