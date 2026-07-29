# TNDP: GNN + RL za dizajn mreze javnog prevoza, case study Novi Sad

Seminarski rad, tema "Eksperimenti sa neuronskim mrezama 2" (PMF Novi Sad).

GATv2 policy trenirana reinforcement learningom (REINFORCE/PPO) resava Transit
Network Design Problem, po uzoru na Holliday et al. (Transportmetrica B, 2025).
Trening na sintetickim gradovima, validacija na Mandl/Mumford benchmarcima,
primena na graf Novog Sada iz open podataka (OSM, GTFS) uz poredjenje sa
postojecom GSP mrezom.

Kompletna specifikacija: [SPEC_seminarski_TNDP.md](SPEC_seminarski_TNDP.md).

## Struktura

```
tndp/
  core/        # CityGraph, TransitNetwork, passenger assignment i cost
  baselines/   # random, greedy, lokalna pretraga
  synth/       # generator sintetickih gradova (uniform i gravity demand)
  rl/          # MDP env, GATv2 + pointer model, trening, evaluacija, MCTS dekoder
  novisad/     # zoniranje, ivice, gravity demand, GTFS baseline
  experiments/ # skripte eksperimenata
  viz/         # mape i krive
configs/       # yaml konfiguracije
data/benchmarks/  # Mandl i Mumford instance (izvor: RenatoArbex/TransitNetworkDesign)
tests/         # acceptance test na Mandlu i unit testovi
```

## Setup

```
pip install -e .[dev]        # core
pip install -e .[rl,geo]     # RL faza i Novi Sad pipeline
```

## Status

Faza 0: priprema. Ceka se potvrda teme od profesora pre faze 1
(core + passenger assignment + Mandl acceptance test).
