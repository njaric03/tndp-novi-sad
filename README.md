# Dizajn mreže linija javnog prevoza pomoću GNN + RL

Transit Network Design Problem (TNDP): dat je graf grada sa uličnom mrežom i matricom
tražnje putovanja, traži se skup autobuskih linija koji dobro opslužuje putnike uz
razuman trošak operatera. Problem je NP-težak i klasično se rešava metaheuristikama po
gradu; ovde umesto toga graf neuronska mreža uči **heuristiku** na hiljadama sintetičkih
gradova, pa je primenjuje na nov grad u jednom prolazu.

Seminarski rad za predmet Eksperimenti sa neuronskim mrežama 2 (DMI, UNSPMF), po uzoru
na Holliday et al. Mreža je GATv2 sa pointer mehanizmom, trenirana REINFORCE-om;
poređenje je sa random search, konstruktivnim greedy-jem i lokalnom pretragom, na
held-out sintetici i na benchmark instancama iz literature (Mandl, Mumford).

> **Napomena o obimu.** Jednoprolazna konstrukcija je *komponenta* metode iz reference,
> ne cela metoda: kod Hollidaya naučena politika radi kao operator unutar metaheuristike,
> i autori mere do 20% razlike u korist hibrida (Neural BCO, arXiv:2306.00720). Ovde je
> `hill_climb` uključen kao baseline upravo da bi ta razlika bila vidljiva.

```
 ┌──────────────────────┐        ┌────────────────────────┐
 │ sintetički generator │        │  benchmark instance    │
 │ Delaunay + gravity   │        │  Mandl, Mumford (CSV)  │
 └──────────┬───────────┘        └───────────┬────────────┘
            │                                │
            └────────────┬───────────────────┘
                         │  CityGraph (coords, street_time, demand)
          ┌──────────────▼───────────────┐
          │  MDP: linija se gradi čvor   │
          │  po čvor, maskirane akcije   │
          └──────────────┬───────────────┘
                         │
          ┌──────────────▼───────────────┐
          │  GATv2 encoder + pointer     │
          │  + halt glava + value glava  │
          └──────────────┬───────────────┘
                         │  REINFORCE (naučen ili greedy baseline)
          ┌──────────────▼───────────────┐
          │  dekodiranje: greedy /       │
          │  sampling k / MCTS (PUCT)    │
          └──────────────┬───────────────┘
                         │
          ┌──────────────▼───────────────┐
          │  passenger assignment → cost │
          │  poređenje sa baselinima     │
          └──────────────────────────────┘
```

## Kako radi

### Graf grada i funkcija cilja

[city.py](tndp/core/city.py) drži `CityGraph`: koordinate čvorova, matricu vremena vožnje
ulicom (`inf` gde nema ivice) i matricu tražnje. [assignment.py](tndp/core/assignment.py)
za datu mrežu linija računa najkraće vreme putovanja svakog para preko grafa linija sa
penalom od 5 min po presedanju, pa iz toga:

- `C_p` — prosečno vreme putovanja, **samo nad opsluženim parovima**. Ovo je konvencija
  u kojoj su objavljeni brojevi iz literature, pa se koristi za poređenje sa njima.
  Između metoda sa različitim `d_un` **nije uporediv**: metoda koja ispusti više parova
  ispušta baš najduže i time sebi ulepšava `C_p`.
- `C_p_all` — isti prosek nad **svim** parovima, gde nepokriven par plaća
  `UNSERVED_FACTOR` puta ulično najkraće vreme. Faktor je 20/5 = 4, odnos brzine autobusa
  i pešaka: putnik bez linije pređe istu razdaljinu pešice. Ovo je metrika po kojoj se
  porede metode.
- `C_o` — ukupno vreme vožnje svih linija u jednom smeru.
- `d_0/d_1/d_2/d_un` — udeli tražnje bez presedanja, sa jednim, sa dva, i nepokrivene.

Funkcija cilja je

```
alpha * C_p_all / (donja granica C_p)  +  (1 - alpha) * C_o / (vreme MST-a)
```

Obe skale su **donje granice iste vrste**: demand-ponderisano najkraće vreme ulicom (kao
da mreža ide svuda) i ukupno vreme minimalnog razapinjućeg stabla (najmanje mreže koliko
treba da svaki čvor bude dostupan). Zato je vrednost ~1 kad je mreža blizu teorijskog
poda, i zato `alpha` zaista balansira — mereno preko kandidat-rešenja, odnos rasipanja
ta dva člana je ~1.4:1. **Nema zasebne kazne za nepokrivenu tražnju**: ona je već u
`C_p_all`. **Ista funkcija se koristi u RL nagradi i u baseline cilju.**

Dve konstante ostaju stvar izbora — `UNSERVED_FACTOR` i `alpha`. Osetljivost na obe
ispisuje `python -m tools.metodoloske_provere` i **treba da ide uz rezultate**.

### Sintetički gradovi

[generator.py](tndp/synth/generator.py) baca slučajne tačke, povezuje ih Delaunay
triangulacijom, izbacuje predugačke ivice i proređuje ostatak do realistične gustine
ulica. Ivica se skida samo ako graf ostane povezan, i grad se na kraju validira — nepovezan
grad ima beskonačnu donju granicu putničkog troška i ne sme da izađe iz generatora.
Tražnja ima dva režima: `uniform` (U[60, 800] po paru, replicira Holliday) i `gravity`
(mase čvorova iz lognormalne, opadanje sa daljinom `1/d^beta`) — gravity je glavni režim.
Ukupan broj putovanja je isti u oba, da su uporedivi.

### Baselines

- [random_search.py](tndp/baselines/random_search.py) — najbolja od *k* nasumičnih mreža.
  Ovo je donja granica, ne ozbiljan takmac.
- [greedy.py](tndp/baselines/greedy.py) — kandidati su najkraći ulični putevi svih parova,
  u svakoj iteraciji se dodaje onaj koji najviše popravlja cilj.
- [hill_climb.py](tndp/baselines/hill_climb.py) — lokalna pretraga nad kompletnim mrežama
  (produži/skrati/zameni liniju), sa restartima. Ovo je ono što u literaturi radi
  metaheuristika i jedini baseline koji stvarno pretražuje.

### MDP i model

[env.py](tndp/rl/env.py): epizoda gradi svih *R* linija redom. Za svaku liniju bira se
početni čvor (dozvoljen terminal), pa se naizmenično bira proširenje ili `halt` kad je
dužina u dozvoljenom opsegu. **Akcija proširenja je par (kraj, čvor)**, ne samo čvor —
inače za čvor susedan oba kraja varijanta „na početak" nije dostižna, a takvih poteza je
na Delaunay grafovima ~13%. Halt je maskiran ako bi linija bila duplikat već sagrađene.

[model.py](tndp/rl/model.py): 13 atributa po čvoru (normalizovane koordinate, produkcija i
atrakcija tražnje, stepen, koncentracija tražnje, pokrivenost, pripadnost tekućoj liniji,
da li je početak, da li je rep, napredak epizode, popunjenost linije, `alpha`), ulične
ivice u oba smera sa vremenom i tražnjom kao edge feature.

Tražnja ulazi u mrežu kroz **rang transformaciju** (`rang → N(0,1)`), ne kao sirov udeo.
U gravity režimu je lognormalna, pa je sirov feature imao asimetriju ~5.3 na ivicama i
raspon 40x — a na Mandlu i Mumfordu asimetriju ~0. Trening i test su time bili različite
raspodele, što je za model čija je poenta transfer veći problem od same skale. Posle
rang transformacije raspodela je ista na svakoj instanci. Apsolutni odnosi koje rang
briše vraćeni su kao jedan skalar po gradu (`concentration`, udeo tražnje u top 10%
parova). **Sirova tražnja ulazi u funkciju cilja nedirnuta** — transformiše se samo ulaz
u mrežu. Tri sloja GATv2 daju embeddinge; pointer glava
skorira parove (kraj, čvor) uslovljeno embeddingom tog kraja, posebna glava skorira
`halt`, value glava daje baseline.

[train.py](tndp/rl/train.py): REINFORCE sa naučenim baseline-om (value glava), opciono
self-critical (greedy rollout, Kool et al.), sa standardizacijom advantage-a unutar
batch-a. **`alpha` se uzorkuje `U[0,1]` po epizodi** (kao kod Hollidaya), pa jedna
politika pokriva ceo Pareto front umesto samo `alpha=0.5`. Trening ide na fiksnom poolu
sintetičkih gradova. Čuvaju se dva checkpointa: `policy.pt` (poslednji) i `best.pt`
(najbolji na validaciji) — evaluacija treba da koristi `best.pt`.

```bash
python -m tndp.rl.train --config configs/rl_default.yaml          # runs/<ime>/best.pt
python -m tndp.rl.train --config configs/rl_default.yaml --seed 1 # drugi seed
```

### Dekodiranje

Ista politika se dekodira na tri načina ([evaluate.py](tndp/rl/evaluate.py),
[mcts.py](tndp/rl/mcts.py)): greedy (argmax), sampling *k* pa najbolja epizoda, i MCTS sa
naučenim priorima (PUCT) po uzoru na AlphaTransit.

Dve svesne razlike od AlphaTransita: tamo vrednost lista daje value mreža i MCTS se koristi
i u treningu (pa value glava uči na međustanjima), a ovde je value glava trenirana samo na
početnom stanju pa je vrednost lista greedy rollout, i pretraga je samo dekoder. Kod njih
se rollout-i izbegavaju jer nagradu daje saobraćajni simulator; ovde je nagrada jedan
Dijkstra na malom grafu. Vrednosti se u stablu normalizuju min-max po stablu, a podstablo
izabrane akcije se zadržava.

## Rezultati

**Rezultati u `results/` su obrisani i treba ih regenerisati** — funkcija cilja,
prostor akcija i generator su se promenili, pa stari brojevi više ne važe i nisu
uporedivi sa novima. Redosled:

```bash
python -m tndp.rl.train --config configs/rl_default.yaml
python -m tndp.experiments.bench_synth     runs/gravity-v1/best.pt   # glavna tabela
python -m tndp.experiments.bench_transfer  runs/gravity-v1/best.pt   # Mandl + Mumford
python -m tndp.experiments.pareto          runs/gravity-v1/best.pt   # Pareto front
python -m tndp.experiments.anytime         runs/gravity-v1/best.pt   # kvalitet vs vreme
python -m tndp.experiments.bench_decoders  runs/gravity-v1/best.pt   # greedy/sampling/MCTS
python -m tndp.experiments.show_networks   runs/gravity-v1/best.pt   # slika mreža
python -m tndp.viz.curves                  runs/gravity-v1           # kriva treninga
```

Tabele nose standardnu devijaciju po gradovima i **uparene** razlike u odnosu na
referentnu metodu (Wilcoxon, iste instance), jer se gradovi po težini razlikuju mnogo više
nego metode među sobom. Svaki eksperiment validira mrežu koju metoda vrati i pada ako
prekrši ograničenja.

[bench_mandl.md](results/bench_mandl.md) i dalje služi kao provera implementacije
assignment-a naspram objavljenih vrednosti, ne kao poređenje metoda.

Ablacije (`value` vs self-critical baseline, fiksni vs uzorkovani `alpha`, gravity vs
uniform tražnja, standardizacija advantage-a) su u [configs/](configs) kao `abl_*.yaml`.

## Pokretanje

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]        # core: numpy, scipy, matplotlib
pip install -e .[rl]         # torch, torch-geometric (trening i evaluacija)
pytest                       # Mandl acceptance + toy assignment + smoke
python -m tools.metodoloske_provere   # invarijante i osetljivost na konstante
```

## Struktura

```
tndp/
  core/        CityGraph, TransitNetwork, passenger assignment i cost
  baselines/   random search, greedy, hill climbing
  synth/       generator sintetičkih gradova (uniform i gravity demand)
  rl/          MDP env, GATv2 + pointer model, REINFORCE trening, dekoderi, MCTS
  novisad/     preuzimanje i sređivanje podataka o Novom Sadu; zoniranje i graf tek dolaze
  experiments/ skripte koje proizvode tabele i slike u results/
  viz/         mape mreža i krive treninga
configs/       yaml konfiguracije treninga i ablacija
data/benchmarks/  Mandl i Mumford instance (izvor: RenatoArbex/TransitNetworkDesign)
data/novisad/  podaci o Novom Sadu (gitignore; pravi se skriptama iz tndp/novisad/)
results/       tabele i slike koje se predaju
tests/         acceptance test na Mandlu, toy assignment, smoke
tools/         provere invarijanti i osetljivosti
docs/          metodološka procena i opis podataka za Novi Sad
```

## Status

Sintetika, baselines, RL trening i evaluacija su gotovi; rezultate treba pregenerisati
posle izmena funkcije cilja. Podaci za Novi Sad su prikupljeni i sređeni — mreža linija
i stajališta ЈГСП-а, granice mesnih zajednica, stanovništvo po zonama i merena
opterećenja linija iz brojanja 2017. Ostaje zoniranje, gravity matrica kalibrisana na
ta opterećenja, ulična mreža preko `osmnx` i poređenje sa postojećom GSP mrežom.

```bash
python -m tndp.novisad.preuzmi && python -m tndp.novisad.sredi
```

[docs/metodoloska-procena.md](docs/metodoloska-procena.md) drži pregled slabih tačaka
eksperimentalnog dizajna, šta je od toga popravljeno i šta ostaje.
[docs/novi-sad.md](docs/novi-sad.md) opisuje svaki izvor podataka za studiju slučaja,
šta je pouzdano i šta je odbačeno.

## Ograničenja modela

Rešava se TRNDP, ne pun TNDP: **nema frekvencija, vremena čekanja, kapaciteta ni veličine
voznog parka**. `C_o` (ukupno vreme vožnje linija u jednom smeru) je *proxy* za trošak
operatera, ne trošak operatera. Presedanje se naplaćuje fiksnih 5 min bez obzira na
frekvenciju linije. Tražnja je statična i neelastična — mreža ne menja to koliko se
putuje. Sve su to standardne pretpostavke Mandl/Mumford benchmark postavke, ali ih treba
imati u vidu pri tumačenju brojeva.

## Izvori

| Uloga | Izvor |
|---|---|
| Metoda (GAT + RL), verzija koju pratimo | Holliday, El-Geneidy, Dudek, *Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning*, https://arxiv.org/abs/2404.05894 (REINFORCE) |
| Objavljena verzija iste metode | Transportmetrica B 13(1), 2025, https://doi.org/10.1080/21680566.2025.2561863 — trenira PPO-om i kombinuje sa evolutivnim algoritmom; mi pratimo raniju, jednostavniju varijantu |
| MDP formulacija i trening | Holliday, *Applications of deep reinforcement learning to urban transit network design*, doktorska teza, https://arxiv.org/abs/2502.17758 |
| Politika kao operator u metaheuristici | Holliday, Dudek, *Neural Bee Colony Optimization*, https://arxiv.org/abs/2306.00720; *A Neural-Evolutionary Algorithm for Autonomous Transit Network Design*, ICRA 2024, https://arxiv.org/abs/2403.07917 |
| MCTS dekodiranje | *AlphaTransit: Learning to Design City-scale Transit Routes*, https://arxiv.org/abs/2605.28730; prethodnik: https://arxiv.org/abs/2512.19767 |
| REINFORCE baseline i sampling dekodiranje | Kool, van Hoof, Welling, *Attention, Learn to Solve Routing Problems!*, ICLR 2019 |
| Benchmark instance | Mumford, *A metaheuristic approach to the urban transit routing problem* (2013); Nikolić, Teodorović, *Transit network design by Bee Colony Optimization* (2013); John, Mumford, Lewis (2014) — konvencija u kojoj računamo `C_p` |
| Fajlovi instanci i objavljena rešenja | https://github.com/RenatoArbex/TransitNetworkDesign |
| Pregled oblasti | *Transit network design problem: a half century of methodological research*, Innovative Infrastructure Solutions (2025), https://doi.org/10.1007/s41062-025-02356-5 |
