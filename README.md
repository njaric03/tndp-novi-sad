# Dizajn mreže linija javnog prevoza pomoću GNN + RL

Transit Network Design Problem (TNDP): dat je graf grada sa uličnom mrežom i matricom
tražnje putovanja, traži se skup autobuskih linija koji dobro opslužuje putnike uz
razuman trošak operatera. Problem je NP-težak i klasično se rešava metaheuristikama po
gradu; ovde graf neuronska mreža uči heuristiku na skupu sintetičkih gradova, pa je
primenjuje na nov grad u jednom prolazu.

Seminarski rad za predmet Eksperimenti sa neuronskim mrežama 2 (DMI, UNSPMF), po uzoru
na Holliday et al. Politika je GATv2 sa pointer mehanizmom, trenirana REINFORCE-om.
Poredi se sa nasumičnom pretragom, konstruktivnom heuristikom i lokalnom pretragom, na
held-out sintetici, na benchmark instancama (Mandl, Mumford) i na Novom Sadu sastavljenom
iz otvorenih podataka.

Ceo argument, brojevi i ograničenja su u samom radu (`seminarski/main.pdf`). Ovaj fajl
opisuje samo kako se repo koristi.

## Funkcija cilja

[assignment.py](tndp/core/assignment.py) za datu mrežu linija računa najkraće vreme
putovanja svakog para preko grafa linija, sa penalom od 5 min po presedanju, pa iz toga:

- `C_p_all`, prosečno vreme putovanja nad svim parovima, gde nepokriven par plaća
  `UNSERVED_FACTOR = 8` puta ulično najkraće vreme. Ovo je metrika po kojoj se porede metode.
- `C_p`, isti prosek samo nad opsluženim parovima. Konvencija iz literature, služi za
  poređenje sa objavljenim brojevima; između metoda sa različitim `d_un` nije uporediv.
- `C_o`, ukupno vreme vožnje svih linija u jednom smeru.
- `d_0/d_1/d_2/d_un`, udeli tražnje bez presedanja, sa jednim, sa dva, i nepokrivene.

Cilj je

```
alpha * C_p_all / (donja granica C_p)  +  (1 - alpha) * C_o / (vreme MST-a)
```

Obe skale su donje granice iste vrste, pa je vrednost oko 1 kad je mreža blizu teorijskog
poda. Ista funkcija se koristi u RL nagradi i u cilju klasičnih metoda, inače poređenje
ne bi merilo metode nego razliku u zadatku.

Skale nisu jednako osetljive: putnički član rasipa 2.9 puta više od operaterskog, pa
`alpha = 0.5` nije neutralna tačka. `UNSERVED_FACTOR` i `alpha` ostaju stvar izbora;
osetljivost na obe ispisuje `python -m tndp.experiments.checks` i treba da ide uz rezultate.

## Pokretanje

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]        # core: numpy, scipy, matplotlib
pip install -e .[rl]         # torch, torch-geometric (trening i evaluacija)
pip install -e .[geo]        # osmnx, geopandas (samo za Novi Sad pipeline)
pip install -e .[nb]         # notebooks/, opciono
pytest
python -m tndp.experiments.checks   # invarijante i osetljivost na konstante
```

### Trening

```bash
python -m tndp.rl.train --config configs/gravity-v2.yaml          # runs/<ime>/best.pt
python -m tndp.rl.train --config configs/gravity-v1.yaml --seed 1 # drugi seed
```

Čuvaju se dva checkpointa, `policy.pt` (poslednji) i `best.pt` (najbolji na validaciji).
Evaluacija treba da koristi `best.pt`.

### Regenerisanje tabela i slika

Zastavice nisu ukras: podrazumevane vrednosti skripti se razlikuju od onoga čime su
predate tabele pravljene, pa bez njih brojevi ne izlaze isti.

```bash
python -m tndp.experiments.bench_synth    runs/gravity-v2/best.pt --cities 20 --out main-20-v2
python -m tndp.experiments.bench_transfer runs/gravity-v2/best.pt --instances Mandl1 Mumford0 Mumford1 Mumford2 Mumford3
python -m tndp.experiments.pareto         runs/gravity-v2/best.pt              # Pareto front i slika
python -m tndp.experiments.anytime        runs/gravity-v2/best.pt --cities 20  # kvalitet vs budžet
python -m tndp.experiments.hybrid         runs/gravity-v2/best.pt --cities 3   # politika kao start pretrage
python -m tndp.experiments.bench_decoders runs/gravity-v2/best.pt --cities 12  # greedy/sampling/MCTS
python -m tndp.experiments.bench_freq     runs/gravity-v1/best.pt              # frekvencije i flota
python -m tndp.experiments.show_networks  runs/gravity-v2/best.pt              # slika mreža
python -m tndp.experiments.policy         runs/gravity-v2/best.pt              # heatmap politike
python -m tndp.viz.curves                 runs/gravity-v2                      # kriva treninga
```

### Novi Sad

Redosled je obavezan, svaki korak čita ono što je prethodni upisao:

```bash
python -m tndp.novisad.preuzmi     # sirovi izvori u data/novisad/raw/
python -m tndp.novisad.sredi       # stajalista, linije, polasci, stanovništvo
python -m tndp.novisad.zone        # zone.csv, stajalista_zone.csv
python -m tndp.novisad.sadrzaji    # privlacnost.csv, POI iz OpenStreetMap-a
python -m tndp.novisad.ulice       # tau.csv, susedstvo.csv, preko osmnx
python -m tndp.novisad.traznja     # traznja.csv, gravitaciona matrica

python -m tndp.novisad.kalibracija # results/novisad-kalibracija.md
python -m tndp.novisad.frekvencije # results/novisad-frekvencije.md
python -m tndp.novisad.poredjenje  # results/novisad-poredjenje.md
python -m tndp.novisad.karta       # karte i results/novisad-struktura.md
```

`data/novisad/` nije u repou, pravi se ovim skriptama.

Trening za studiju slučaja ne traži ništa od ovoga. `configs/novisad-r19.yaml`
uči na sintetičkim gradovima veličine Novog Sada, a podaci gore služe tek
evaluaciji, pa ta dva posla mogu da idu uporedo.

## Rezultati

Model iz glavnih tabela je `runs/gravity-v2` (sintetika, benchmark instance) i
`runs/novisad-r19` (Novi Sad). Dva izuzetka, oba označena u zaglavlju samog fajla:
ablacije su na `runs/gravity-v1` jer se porede međusobno na kraćem rasporedu, a
[bench-freq.md](results/bench-freq.md) je takođe još na `gravity-v1`.

Tabele nose standardnu devijaciju po gradovima i uparene razlike u odnosu na referentnu
metodu (Wilcoxon, iste instance), jer se gradovi po težini razlikuju mnogo više nego
metode među sobom. [bench-mandl.md](results/bench-mandl.md) služi kao provera
implementacije assignment-a naspram objavljenih vrednosti, ne kao poređenje metoda.

Glavne brojke: [main-20-v2.md](results/main-20-v2.md) (sintetika),
[pareto.md](results/pareto.md) (front po alfi), [hybrid.md](results/hybrid.md) (spoj),
[bench-transfer.md](results/bench-transfer.md) (Mandl i Mumford),
[novisad-poredjenje.md](results/novisad-poredjenje.md) (Novi Sad).

### Testovi

| fajl | šta proverava |
|---|---|
| `test_assignment.py` | dodela putnika na ruke: vremena, presedanja, nepokriveni parovi |
| `test_frequencies.py` | intervali, flota, skale i petlja druge faze, sve na ruke |
| `test_mandl_acceptance.py` | naša cost funkcija naspram objavljenih Mandl vrednosti |

Pokrivaju jedini deo koda gde se greška ne vidi kao pad nego kao pogrešan broj. Ostalo se
proverava pokretanjem samih skripti, koje padaju ako metoda vrati nevalidnu mrežu.

### Reproducibilnost

```
Python 3.11   torch 2.13.0+cpu   torch-geometric 2.8.0.post1
numpy 2.4.6   scipy 1.17.1       matplotlib 3.11.1
```

Klasične metode i `RL greedy dekod` su deterministični. Redovi `RL sampling 32` nisu:
dekoder uzima najbolju od 32 uzorkovane epizode, pa razlika u poslednjim bitovima GATv2
izlaza, koju donosi druga verzija `torch-geometric`, ume da promeni koja epizoda pobedi.
Na proseku preko 20 gradova to je ±0.002, na pojedinačnoj instanci do ±0.03.

## Struktura

```
tndp/
  core/        CityGraph, TransitNetwork, passenger assignment, cost, frekvencije,
               učitavanje instanci i generator sintetičkih gradova (synth.py)
  baselines/   nasumična pretraga, konstruktivna heuristika, lokalna pretraga
  rl/          MDP env, GATv2 + pointer model, REINFORCE trening, dekoderi, MCTS
  novisad/     preuzimanje i sređivanje podataka, zoniranje, tražnja, kalibracija, karte
  experiments/ skripte koje pokreću metode i modele i proizvode results/
  viz/         crtanje: stil, karte mreža, krive treninga, figure za rad
configs/       yaml konfiguracije treninga i ablacija
data/benchmarks/  Mandl i Mumford instance (izvor: RenatoArbex/TransitNetworkDesign)
results/       tabele i slike koje se predaju
notebooks/     tri sveske koje pozivaju postojeći kod: podaci, model, rezultati
tests/         acceptance test na Mandlu, assignment i frekvencije na ruke
```

Dve konvencije važe kroz ceo repo. Imena u kodu su engleska svuda osim u `tndp/novisad/`,
koji je domenski paket pa se u njemu sve zove kao u samim izvorima podataka (`zone`,
`stajalista`, `polasci`). Komentari su srpski bez dijakritike, dok tekst koji čitalac
vidi, dakle oznake na slikama i izveštaji u `results/`, ide sa dijakritikom.

## Izvori

| Uloga | Izvor |
|---|---|
| Metoda (GAT + RL), verzija koju pratimo | Holliday, El-Geneidy, Dudek, *Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning*, https://arxiv.org/abs/2404.05894 |
| Objavljena verzija iste metode | Transportmetrica B 13(1), 2025, https://doi.org/10.1080/21680566.2025.2561863 |
| MDP formulacija i trening | Holliday, *Applications of deep reinforcement learning to urban transit network design*, doktorska teza, https://arxiv.org/abs/2502.17758 |
| Politika kao operator u metaheuristici | Holliday, Dudek, *Neural Bee Colony Optimization*, https://arxiv.org/abs/2306.00720; *A Neural-Evolutionary Algorithm for Autonomous Transit Network Design*, ICRA 2024, https://arxiv.org/abs/2403.07917 |
| MCTS dekodiranje | *AlphaTransit: Learning to Design City-scale Transit Routes*, https://arxiv.org/abs/2605.28730 |
| REINFORCE baseline i sampling dekodiranje | Kool, van Hoof, Welling, *Attention, Learn to Solve Routing Problems!*, ICLR 2019 |
| Benchmark instance | Mumford (2013); Nikolić, Teodorović (2013); John, Mumford, Lewis (2014) |
| Fajlovi instanci i objavljena rešenja | https://github.com/RenatoArbex/TransitNetworkDesign |
| Brojanje putnika po linijama, Novi Sad 2017 | Lazarević et al. (2020) |
| Pregled oblasti | *Transit network design problem: a half century of methodological research*, Innovative Infrastructure Solutions (2025), https://doi.org/10.1007/s41062-025-02356-5 |
