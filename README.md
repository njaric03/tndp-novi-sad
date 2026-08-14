# Dizajn mreže linija javnog prevoza pomoću GNN + RL

Transit Network Design Problem (TNDP): dat je graf grada sa uličnom mrežom i matricom
tražnje putovanja, traži se skup autobuskih linija koji dobro opslužuje putnike uz
razuman trošak operatera. Problem je NP-težak i klasično se rešava metaheuristikama po
gradu; ovde umesto toga graf neuronska mreža uči heuristiku na skupu sintetičkih
gradova, pa je primenjuje na nov grad u jednom prolazu.

Seminarski rad za predmet Eksperimenti sa neuronskim mrežama 2 (DMI, UNSPMF), po uzoru
na Holliday et al. Politika je GATv2 sa pointer mehanizmom, trenirana REINFORCE-om.
Poredi se sa nasumičnom pretragom, konstruktivnom heuristikom i lokalnom pretragom, na
held-out sintetici, na benchmark instancama iz literature (Mandl, Mumford) i na Novom
Sadu sastavljenom iz otvorenih podataka.

Jednoprolazna konstrukcija je komponenta metode iz reference, ne cela metoda: kod
Hollidaya naučena politika radi kao operator unutar metaheuristike, i tek taj spoj daje
najbolje objavljene rezultate. Zato je lokalna pretraga ovde uključena kao ravnopravan
protivnik, a spoj sa njom je i sam meren (`hybrid.py`).

```
 ┌──────────────────────┐        ┌────────────────────────┐
 │ sintetički generator │        │  benchmark instance    │
 │ Delaunay + gravity   │        │  Mandl, Mumford (CSV)  │
 └──────────┬───────────┘        └───────────┬────────────┘
            │                                │
            │        ┌───────────────────────┴────┐
            │        │  Novi Sad: OSM, red vožnje │
            │        │  brojanje putnika 2017     │
            │        └───────────────────────┬────┘
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

- `C_p`, prosečno vreme putovanja samo nad opsluženim parovima. To je konvencija u kojoj
  su objavljeni brojevi iz literature, pa se koristi za poređenje sa njima. Između metoda
  sa različitim `d_un` nije uporediv: metoda koja ispusti više parova ispušta baš najduže
  i time sebi ulepšava `C_p`.
- `C_p_all`, isti prosek nad svim parovima, gde nepokriven par plaća `UNSERVED_FACTOR`
  puta ulično najkraće vreme. Faktor je 8: četiri puta je odnos brzina autobusa i pešaka
  (20 naspram 5 km/h), a još dvaput jer se minut pešačenja doživljava kao dva minuta
  vožnje. Ovo je metrika po kojoj se porede metode.

  Faktor 8 **nije** iznad najgoreg opsluženog para. Mereno na izmerenim mrežama
  (`python -m tndp.experiments.checks`, provera 2), opslužen par putuje 1.9 puta duže
  od uličnog najkraćeg u demand-ponderisanom proseku, 5.3 puta na 95. percentilu, a
  maksimum ide do 20.9 puta. Faktor 8 je iznad 98% opslužene tražnje; preostala 2%
  su parovi koje se optimizatoru i dalje isplati ispustiti umesto opslužiti. To je
  poznato ograničenje cilja, ne rešen problem: faktor iznad ~20 bi ga zatvorio, ali
  bi mrežu naterao da pokrije sve po svaku cenu. Sweep po faktoru od 1.5 do 8
  ispisuje ista provera i treba da ide uz rezultate.
- `C_o`, ukupno vreme vožnje svih linija u jednom smeru.
- `d_0/d_1/d_2/d_un`, udeli tražnje bez presedanja, sa jednim, sa dva, i nepokrivene.

Funkcija cilja je

```
alpha * C_p_all / (donja granica C_p)  +  (1 - alpha) * C_o / (vreme MST-a)
```

Obe skale su donje granice iste vrste: demand-ponderisano najkraće vreme ulicom (kao da
mreža ide svuda) i ukupno vreme minimalnog razapinjućeg stabla (najmanja mreža koliko
treba da svaki čvor bude dostupan). Zato je vrednost oko 1 kad je mreža blizu teorijskog
poda. Nema zasebne kazne za nepokrivenu tražnju, ona je već u `C_p_all`. Ista funkcija se
koristi u RL nagradi i u cilju klasičnih metoda, inače poređenje ne bi merilo metode nego
razliku u zadatku.

Skale jesu iste vrste, ali **nisu jednako osetljive**. Mereno preko kandidat-rešenja
(`checks.py`, provera 1), putnički član rasipa 2.9 puta više od operaterskog. Posledica
je da `alpha = 0.5` nije neutralna tačka: jednak uticaj oba člana je oko `alpha ≈ 0.25`,
a pri `alpha = 0.5` cilj prati uglavnom putnički član. To treba imati u vidu pri čitanju
svake tabele na `alpha = 0.5` i objašnjava zašto se poredak metoda menja duž fronta
([pareto.md](results/pareto.md)).

Dve konstante ostaju stvar izbora, `UNSERVED_FACTOR` i `alpha`. Osetljivost na obe
ispisuje `python -m tndp.experiments.checks` i treba da ide uz rezultate.

### Sintetički gradovi

[synth.py](tndp/synth.py) baca slučajne tačke, povezuje ih Delaunay triangulacijom,
izbacuje predugačke ivice i proređuje ostatak do realistične gustine ulica. Ivica se
skida samo ako graf ostane povezan, i grad se na kraju validira: nepovezan grad ima
beskonačnu donju granicu putničkog troška, pa putnički deo cilja tiho nestane bez ijedne
poruke o grešci. Tražnja ima dva režima, `uniform` (U[60, 800] po paru, replicira
Holliday) i `gravity` (mase čvorova iz lognormalne, opadanje sa daljinom `1/d^beta`).
Gravity je glavni režim, uniform postoji kao ablacija. Ukupan broj putovanja je isti u
oba, da su uporedivi.

### Metode za poređenje

- [random_search.py](tndp/baselines/random_search.py), najbolja od *k* nasumičnih mreža.
  Donja granica, ne ozbiljan takmac.
- [greedy.py](tndp/baselines/greedy.py), konstruktivna heuristika: kandidati su najkraći
  ulični putevi svih parova, u svakoj iteraciji se dodaje onaj koji najviše popravlja cilj.
- [hill_climb.py](tndp/baselines/hill_climb.py), lokalna pretraga nad kompletnim mrežama
  (produži, skrati ili zameni liniju), sa restartima. To je ono što u literaturi radi
  metaheuristika i jedina metoda koja stvarno pretražuje.

Sve tri optimizuju istu funkciju cilja i sa istim `alpha` kao politika.

### MDP i model

[env.py](tndp/rl/env.py): epizoda gradi svih *R* linija redom. Za svaku liniju bira se
početni čvor, pa se naizmenično bira proširenje ili `halt` kad je dužina u dozvoljenom
opsegu. Akcija proširenja je par (kraj, čvor), ne samo čvor, jer linija raste na obe
strane; inače za čvor susedan oba kraja varijanta na početak nije dostižna. Na Delaunay
grafovima 40% poteza nudi bar jedan takav čvor (`checks.py`, provera 8), a među samim
dozvoljenim akcijama dvosmisleno je oko 10%. Nedozvoljeni potezi se maskiraju umesto da
se kažnjavaju kroz nagradu, pa politika ne troši trening na pravila koja su ionako
poznata.
Halt je maskiran i ako bi linija bila duplikat već sagrađene.

[features.py](tndp/rl/features.py): 13 atributa po čvoru (normalizovane koordinate, produkcija
i atrakcija tražnje, stepen, koncentracija tražnje, pokrivenost, pripadnost tekućoj
liniji, da li je početak, da li je rep, napredak epizode, popunjenost linije, `alpha`),
ulične ivice u oba smera sa vremenom i tražnjom kao edge feature.

Tražnja ulazi u mrežu kroz rang transformaciju (`rang → N(0,1)`), ne kao sirov udeo. U
gravity režimu je lognormalna, pa je sirov feature imao asimetriju oko 5.3 na ivicama i
raspon 40x, a na Mandlu i Mumfordu asimetriju oko nule. Trening i test su time bili
različite raspodele, što je za model čija je poenta transfer veći problem od same skale.
Posle rang transformacije raspodela je ista na svakoj instanci. Apsolutni odnosi koje
rang briše vraćeni su kao jedan skalar po gradu (`concentration`, udeo tražnje u top 10%
parova). Sirova tražnja ulazi u funkciju cilja nedirnuta, transformiše se samo ulaz u
mrežu. [model.py](tndp/rl/model.py): tri sloja GATv2 daju embeddinge; pointer glava
skorira parove (kraj, čvor) uslovljeno embeddingom tog kraja, posebna glava skorira
`halt`, value glava daje baseline.

[train.py](tndp/rl/train.py): REINFORCE sa naučenim baseline-om (value glava), opciono
self-critical (greedy rollout, Kool et al.), sa standardizacijom advantage-a unutar
batch-a. Vrednost `alpha` se uzorkuje `U[0,1]` po epizodi (kao kod Hollidaya), pa jedna
politika pokriva ceo Pareto front umesto samo `alpha=0.5`. Trening ide na fiksnom poolu
od 512 sintetičkih gradova; validacioni i test gradovi dolaze iz odvojenih semena.
Čuvaju se dva checkpointa, `policy.pt` (poslednji) i `best.pt` (najbolji na validaciji);
evaluacija treba da koristi `best.pt`, jer je kod ovoliko šumovitog treninga razlika
osetna.

```bash
python -m tndp.rl.train --config configs/gravity-v2.yaml          # runs/<ime>/best.pt
python -m tndp.rl.train --config configs/gravity-v1.yaml --seed 1 # drugi seed
```

### Dekodiranje

Ista politika se dekodira na tri načina ([evaluate.py](tndp/rl/evaluate.py),
[mcts.py](tndp/rl/mcts.py)): greedy (argmax), sampling *k* pa najbolja epizoda, i MCTS sa
naučenim priorima (PUCT) po uzoru na AlphaTransit. Sva tri koriste isti model i
razlikuju se samo po utrošenom računanju.

Dve svesne razlike od AlphaTransita: tamo vrednost lista daje value mreža i MCTS se koristi
i u treningu (pa value glava uči na međustanjima), a ovde je value glava trenirana samo na
početnom stanju pa je vrednost lista greedy rollout, i pretraga je samo dekoder. Kod njih
se rollout-i izbegavaju jer nagradu daje saobraćajni simulator; ovde je nagrada jedan
Dijkstra na malom grafu. Vrednosti se u stablu normalizuju min-max po stablu, a podstablo
izabrane akcije se zadržava.

Na 12 gradova ([bench-decoders.md](results/bench-decoders.md)) sampling 32 daje 1.664,
MCTS sa 50 simulacija po potezu 1.778, a greedy dekodiranje 2.010, uz oko tri puta veće
vreme MCTS-a od samplinga. Pretraga stabla je dakle konkurentna ali ne tuče sampling, pa
je glavni dekoder u svim tabelama sampling 32.

### Novi Sad

[tndp/novisad/](tndp/novisad) sastavlja instancu iz otvorenih podataka: `preuzmi.py` i
`sredi.py` skidaju i čiste izvore, `zone.py` deli grad na 32 mesne zajednice kroz koje
prolazi bar jedna gradska linija, `ulice.py` daje vremena vožnje preko OSM ulične mreže,
`traznja.py` gravitacionu matricu, `instanca.py` prevodi GSP mrežu u `TransitNetwork`.

`kalibracija.py` traži par (beta, prag pešačenja) pri kom postojeća mreža reprodukuje
brojanje putnika iz 2017. Prag jeste određen i iznosi 3.5 km, jer ukupno varijaciono
rastojanje profila ima jasan minimum tu; beta nije, jer je pri tom pragu rastojanje
jednako na tri decimale za svaku vrednost. Usvojeno je `beta = 2.0`, ista vrednost koju
koristi generator, da se raspodela tražnje na Novom Sadu ne razlikuje od one na kojoj je
politika trenirana.

`frekvencije.py` pušta frekvencijsku fazu na postojeću mrežu, čiji je red vožnje poznat,
a model o njemu ne dobija nijedan podatak. To je jedina spoljna provera u repou koja se
ne oslanja na podatak koji je i sam ušao u model.

```bash
python -m tndp.novisad.preuzmi && python -m tndp.novisad.sredi
python -m tndp.novisad.kalibracija     # results/novisad-kalibracija.md
python -m tndp.novisad.frekvencije     # results/novisad-frekvencije.md
python -m tndp.novisad.poredjenje      # results/novisad-poredjenje.md
python -m tndp.novisad.karta           # karte i results/novisad-struktura.md
```

## Rezultati

Model iz glavnih tabela je `runs/gravity-v2` (sintetika, benchmark instance) i
`runs/novisad-r19` (Novi Sad). Dva izuzetka, oba označena u zaglavlju samog fajla:
ablacije su na `runs/gravity-v1` jer se porede međusobno na kraćem rasporedu od 3000
iteracija, a [bench-freq.md](results/bench-freq.md) je takođe još na `gravity-v1` i
čeka ponovno pokretanje. Tabele nose standardnu devijaciju po gradovima i uparene
razlike u odnosu na referentnu metodu (Wilcoxon, iste instance), jer se gradovi po težini
razlikuju mnogo više nego metode među sobom. Svaki eksperiment validira mrežu koju metoda
vrati i pada ako prekrši ograničenja.

Redosled kojim se sve regeneriše. Zastavice nisu ukras: podrazumevane vrednosti skripti
se razlikuju od onoga čime su predate tabele pravljene, pa bez njih brojevi ne izlaze isti.

```bash
python -m tndp.rl.train --config configs/gravity-v2.yaml
python -m tndp.experiments.bench_synth    runs/gravity-v2/best.pt --cities 20 --out main-20-v2
python -m tndp.experiments.bench_transfer runs/gravity-v2/best.pt --instances Mandl1 Mumford0 Mumford1 Mumford2 Mumford3
python -m tndp.experiments.pareto         runs/gravity-v2/best.pt              # Pareto front i slika
python -m tndp.experiments.anytime        runs/gravity-v2/best.pt --cities 20  # kvalitet vs budžet
python -m tndp.experiments.hybrid         runs/gravity-v2/best.pt --cities 3   # politika kao start pretrage
python -m tndp.experiments.bench_decoders runs/gravity-v2/best.pt --cities 12  # greedy/sampling/MCTS
python -m tndp.experiments.bench_freq     runs/gravity-v1/best.pt              # frekvencije i flota (v1, vidi gore)
python -m tndp.experiments.show_networks  runs/gravity-v2/best.pt              # slika mreža
python -m tndp.viz.policy                 runs/gravity-v2/best.pt              # heatmap politike
python -m tndp.viz.curves                 runs/gravity-v2                      # kriva treninga
```

[bench-mandl.md](results/bench-mandl.md) služi kao provera implementacije assignment-a
naspram objavljenih vrednosti, ne kao poređenje metoda.

### Glavni brojevi

Na 20 held-out gradova, `R=4`, `alpha=0.5` ([main-20-v2.md](results/main-20-v2.md)):
lokalna pretraga 1.474, politika sa uzorkovanjem 32 daje 1.630, konstruktivna heuristika
1.948, greedy dekodiranje politike 2.109, nasumična pretraga 2.149. Politika sa
uzorkovanjem nadmašuje konstruktivnu heuristiku statistički značajno, i gubi od lokalne
pretrage.

Jedna vrednost `alpha` ipak ne opisuje odnos snaga. Na Pareto frontu
([pareto.md](results/pareto.md)) politika vodi nad konstruktivnom heuristikom iznad
`alpha ≈ 0.43`, i razlika raste sa `alpha`; oko `alpha ≈ 0.85` prestiže i lokalnu
pretragu (1.554 naspram 1.575 pri `alpha = 0.9`). Razlog se čita iz pokrivenosti:
politika gradi mreže koje opslužuju tražnju, dok konstruktivna heuristika bira najkraće
ulične puteve i ne stiže do pune pokrivenosti sa `R = 4`. Vrednost `alpha = 0.5` je za
politiku nepovoljna tačka fronta, pa jedan objavljen `alpha` sistematski potcenjuje
naučenu heuristiku.

Najbolji rezultat u repou daje spoj, ne nijedna metoda sama
([hybrid.md](results/hybrid.md)). Lokalna pretraga iz starta koji je dala politika daje
1.448, uz utrošak od 32 evaluacije od 3000 na pravljenje starta; iz slučajnog starta daje
1.568, a iz konstruktivne heuristike 1.602, dakle gore nego iz slučajnog, uz trećinu
budžeta potrošenu na start. Mereno je na 3 grada, pa je nalaz preliminaran, ali smer je
isti na sva tri.

Šta je politika naučila meri [policy-traznja.md](results/policy-traznja.md): Spearmanova
korelacija između verovatnoće da čvor bude izabran kao početni i tražnje u njemu je
+0.973 na svih 20 gradova, a sa stepenom čvora u uličnoj mreži +0.014. Naučeno pravilo za
početak linije je dakle rangiranje po tražnji, a ne po topologiji.

### Varijansa po seed-u

Ista konfiguracija sa tri semena treninga daje 1.877, 2.003 i 2.437, dakle rasipanje od
±0.294, a nijedna ispitana izmena treninga ili ulaza ne pomera rezultat ni približno
toliko ([ablacije-upareno.md](results/ablacije-upareno.md), pojedinačne tabele u
`abl-*.md`, configi u [configs/](configs)). Uparen Wilcoxon daje `p < 0.01` i za dva
semena iste konfiguracije, što znači da značajnost tamo meri razliku između dva treninga,
a ne efekat izmene. Sa jednim semenom po varijanti nijedna ablacija ne nosi zaključak
sama za sebe.

Izvor rasipanja nije nestabilnost učenja nego to gde na frontu seme završi: tri semena
imaju `d_un` 0.082, 0.122 i 0.267 uz `C_o` 69, 64 i 41. Pošto se `alpha` u nagradi
uzorkuje po epizodi, ništa u treningu ne bira gde će politika sesti na frontu, a
`alpha = 0.5` te izbore onda oštro razdvaja.

### Prenos i Novi Sad

Model treniran na gradovima sa 15 do 30 čvorova i `R = 4` pušta se bez dodatnog treniranja
na instance sa do 127 čvorova i 60 linija ([bench-transfer.md](results/bench-transfer.md)).
Zaostatak za lokalnom pretragom je 4% na Mandlu (1.288 naspram 1.241) i 12% na Mumford3
(4.552 naspram 4.081), ali ne prati veličinu monotono: najveći je na Mumford0 (47%), gde
politika jedina ostavlja tražnju bez veze. Van raspodele nije samo veličina nego i
dozvoljena dužina linije ([bench_transfer.py](tndp/experiments/bench_transfer.py),
`INSTANCES`): Mumford1 i Mumford2 traže najmanje 10 čvorova po liniji, Mumford3 najmanje
12, a politika je trenirana na opsegu [2, 8]. Mumford0 to ne objašnjava, njegov opseg je
[2, 15] pa mu je samo gornja granica van trening raspona.

Na Novom Sadu ([novisad-poredjenje.md](results/novisad-poredjenje.md), `n = 32` zone,
`R = 19`) poredak je isti kao na sintetici: lokalna pretraga 1.648, konstruktivna
heuristika 1.681, politika sa uzorkovanjem 2.307, greedy dekodiranje 2.694, nasumična
pretraga 3.142, postojeća GSP mreža 3.313. Da politika nadmašuje GSP nije jaka tvrdnja,
jer postojeća mreža nije projektovana po ovom cilju. Uporediviji je Jaccard po parovima
uzastopnih zona, koji ne zavisi od izbora cilja: nijedna metoda ne rekonstruiše postojeće
koridore naročito verno, a politika najmanje (0.391).

Frekvencijska faza puštena na postojeću mrežu ([novisad-frekvencije.md](results/novisad-frekvencije.md))
daje medijanu apsolutne greške intervala od 7.0 min, Spearmanovu korelaciju sa objavljenim
intervalima +0.401 i vozni park od 88 vozila. Poredak linija model pogađa, pojedinačne
vrednosti ne.

## Pokretanje

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]        # core: numpy, scipy, matplotlib
pip install -e .[rl]         # torch, torch-geometric (trening i evaluacija)
pip install -e .[geo]        # osmnx, geopandas (samo za Novi Sad pipeline)
pytest                       # vidi niže šta pokriva i šta se preskače
pip install -e .[nb]         # notebooks/, opciono
python -m ipykernel install --user --name tndp --display-name "Python (tndp)"
python -m tndp.experiments.checks   # invarijante i osetljivost na konstante
```

### Šta testovi pokrivaju

| fajl | šta proverava | uslov |
|---|---|---|
| `test_assignment.py` | dodela putnika na ruke: vremena, presedanja, nepokriveni parovi | uvek |
| `test_frequencies.py` | intervali, flota, skale i petlja druge faze, sve na ruke | uvek |
| `test_mandl_acceptance.py` | naša cost funkcija naspram objavljenih Mandl vrednosti | uvek |
| `test_regresija.py` | generator, baselines, MDP, politika i sva tri dekodera | RL testovi traže `[rl]` |
| `test_novisad.py` | zonski graf, rekonstruisana GSP mreža, frekvencije na njoj | traži `data/novisad/` |

Dva skupa se **tiho preskaču** ako im uslov nije ispunjen, pa `pytest -rs` pokazuje
šta je stvarno pušteno. Bez `pip install -e .[rl]` cela RL polovina suite-a ne radi, a
`data/novisad/` se ne isporučuje u repou nego se pravi skriptama iz `tndp/novisad/`.

### Okruženje u kom su tabele napravljene

```
Python 3.11   torch 2.13.0+cpu   torch-geometric 2.8.0.post1
numpy 2.4.6   scipy 1.17.1       matplotlib 3.11.1
```

Klasične metode i `RL greedy dekod` su deterministični i reprodukuju se do treće
decimale na svakoj instalaciji. Redovi **`RL sampling 32` nisu tako robusni**:
dekoder uzima najbolju od 32 uzorkovane epizode, pa razlika u poslednjim bitovima
GATv2 izlaza, koju donosi druga verzija `torch-geometric`, ume da promeni koja
epizoda pobedi. Na proseku preko 20 gradova to je ±0.002, ali na pojedinačnoj
instanci ide i do ±0.03. Ako brojevi ne izlaze isti, prvo proveri verzije.

## Struktura

```
tndp/
  core/        CityGraph, TransitNetwork, passenger assignment, cost, frekvencije
  baselines/   nasumična pretraga, konstruktivna heuristika, lokalna pretraga
  synth.py     generator sintetičkih gradova (uniform i gravity tražnja)
  rl/          MDP env, GATv2 + pointer model, REINFORCE trening, dekoderi, MCTS
  novisad/     preuzimanje i sređivanje podataka, zoniranje, tražnja, kalibracija, karte
  experiments/ skripte koje proizvode tabele i slike u results/
  viz/         karte mreža, krive treninga, heatmap politike, figure za rad
configs/       yaml konfiguracije treninga i ablacija
data/benchmarks/  Mandl i Mumford instance (izvor: RenatoArbex/TransitNetworkDesign)
data/novisad/  podaci o Novom Sadu (gitignore, prave se skriptama iz tndp/novisad/)
results/       tabele i slike koje se predaju
notebooks/     tri sveske koje pozivaju postojeći kod: podaci, model, rezultati
tests/         acceptance test na Mandlu, assignment na ruke, regresija
```

Dve konvencije važe kroz ceo repo, da se ne bi razišao po tome kad je koji fajl
pisan. Imena u kodu su engleska svuda osim u `tndp/novisad/`, koji je domenski
paket pa se u njemu sve zove kao u samim izvorima podataka (`zone`, `stajalista`,
`polasci`). Komentari su srpski bez dijakritike, dok tekst koji čitalac vidi,
dakle oznake na slikama i izveštaji u `results/`, ide sa dijakritikom.

## Ograničenja modela

Najozbiljnije ograničenje nije u modelu nego u raspoređivanju putnika. Svaki par zona
bira jedan najbrži put i sva tražnja tog para ide na njega, pa među paralelnim linijama u
istom koridoru pobednik uzima sve. Posledica je merljiva: na mreži Novog Sada devet od
devetnaest linija dobija tačno nula putnika
([novisad-frekvencije.md](results/novisad-frekvencije.md)). Stvarni putnik ulazi u prvu
liniju koja naiđe, pa se opterećenje deli po frekvencijama; taj model dodele po strategiji
ovde nije implementiran i predstavlja gornju granicu tačnosti svakog poređenja po linijama.

Trase se biraju TRNDP ciljem, bez frekvencija, vremena čekanja, kapaciteta i veličine
voznog parka. Frekvencijska faza ([frequencies.py](tndp/core/frequencies.py)) je druga
faza nad gotovom mrežom: iz opterećenja najopterećenije deonice određuje interval sleđenja
tako da vrh stane u kapacitet vozila, iz njega vreme čekanja od pola intervala umesto
fiksnih 5 min, i broj vozila iz vremena obilaska. Zavisnost je kružna, pa se rešava u
nekoliko prolaza. Kroz istu fazu prolaze sve metode, pa poređenje ostaje pošteno, ali
mreža po voznom parku ostaje merena a ne za njega optimizovana.
[bench-freq.md](results/bench-freq.md) daje obe ocene jednu do druge; poredak metoda se
između njih menja, jer trasa koja dobro opslužuje putnike ne mora da bude jeftina za
vozni park. Da frekvencije uđu u samu nagradu, treba im mesto u MDP-u (izbor intervala
kao akcija posle `halt`-a, uz ograničenje ukupne flote) i ponovni trening svega.

Tražnja je statična i neelastična, dakle mreža ne menja koliko se putuje. Tražnja za Novi
Sad je procena, pa sve što se o tom gradu tvrdi važi za njegov model, a poređenje sa
mrežom GSP-a je poređenje dva modela a ne dva sistema. Prag pešačenja od 3.5 km oblikuje
matricu jače nego bilo koja druga odluka u pripremi podataka: on autobusku tražnju
premešta na periferiju, pa i mreže koje optimizator predlaže.

## Izvori

| Uloga | Izvor |
|---|---|
| Metoda (GAT + RL), verzija koju pratimo | Holliday, El-Geneidy, Dudek, *Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning*, https://arxiv.org/abs/2404.05894 (REINFORCE) |
| Objavljena verzija iste metode | Transportmetrica B 13(1), 2025, https://doi.org/10.1080/21680566.2025.2561863, trenira PPO-om i kombinuje sa evolutivnim algoritmom; ovde se prati ranija, jednostavnija varijanta |
| MDP formulacija i trening | Holliday, *Applications of deep reinforcement learning to urban transit network design*, doktorska teza, https://arxiv.org/abs/2502.17758 |
| Politika kao operator u metaheuristici | Holliday, Dudek, *Neural Bee Colony Optimization*, https://arxiv.org/abs/2306.00720; *A Neural-Evolutionary Algorithm for Autonomous Transit Network Design*, ICRA 2024, https://arxiv.org/abs/2403.07917 |
| MCTS dekodiranje | *AlphaTransit: Learning to Design City-scale Transit Routes*, https://arxiv.org/abs/2605.28730; prethodnik: https://arxiv.org/abs/2512.19767 |
| REINFORCE baseline i sampling dekodiranje | Kool, van Hoof, Welling, *Attention, Learn to Solve Routing Problems!*, ICLR 2019 |
| Benchmark instance | Mumford, *A metaheuristic approach to the urban transit routing problem* (2013); Nikolić, Teodorović, *Transit network design by Bee Colony Optimization* (2013); John, Mumford, Lewis (2014), konvencija u kojoj se računa `C_p` |
| Fajlovi instanci i objavljena rešenja | https://github.com/RenatoArbex/TransitNetworkDesign |
| Brojanje putnika po linijama, Novi Sad 2017 | Lazarević et al. (2020) |
| Pregled oblasti | *Transit network design problem: a half century of methodological research*, Innovative Infrastructure Solutions (2025), https://doi.org/10.1007/s41062-025-02356-5 |
