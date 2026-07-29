import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from tndp.rl.env import HALT


# feature vektor po čvoru za trenutno stanje epizode
def node_features(env):
    city = env.city
    n = city.n
    coords = (city.coords - city.coords.mean(0)) / (city.coords.std(0) + 1e-6)
    dem_out = city.demand.sum(1) / city.demand.sum()
    dem_in = city.demand.sum(0) / city.demand.sum()
    degree = np.array([len(nb) for nb in env.neighbors]) / 4.0
    covered = np.zeros(n)
    for r in env.routes:
        covered[r] = 1.0
    in_current = np.zeros(n)
    is_end = np.zeros(n)
    if env.current:
        in_current[env.current] = 1.0
        is_end[[env.current[0], env.current[-1]]] = 1.0
    progress = len(env.routes) / env.num_routes
    x = np.column_stack([coords, dem_out * n, dem_in * n, degree, covered,
                         in_current, is_end,
                         np.full(n, progress), np.full(n, env.alpha)])
    return torch.tensor(x, dtype=torch.float32)


# ulične ivice u oba smera + tau i demand para kao edge feature
def edge_tensors(city):
    e = city.street_edges
    idx = np.concatenate([e, e[:, ::-1]]).T
    tau = city.street_time[idx[0], idx[1]]
    dem = city.demand[idx[0], idx[1]] * city.n / city.demand.sum()
    attr = np.column_stack([tau / tau.mean(), dem])
    return (torch.tensor(idx, dtype=torch.long),
            torch.tensor(attr, dtype=torch.float32))


# GATv2 encoder + pointer glava za izbor čvora + halt glava + value glava
class TndpPolicy(nn.Module):
    def __init__(self, in_dim=10, hidden=64, layers=3, heads=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=2)
            for _ in range(layers)])
        self.node_score = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.halt_score = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x, edge_index, edge_attr):
        h = self.embed(x)
        for conv in self.convs:
            h = h + torch.relu(conv(h, edge_index, edge_attr))
        return h

    # logiti nad akcijama: čvorovi (maskirani) + opciono halt na kraju
    def action_logits(self, h, decision, mask):
        context = h.mean(0)
        scores = self.node_score(
            torch.cat([h, context.expand(h.shape[0], -1)], dim=1)).squeeze(-1)
        scores = scores.masked_fill(~torch.tensor(mask), -torch.inf)
        if decision == HALT:
            halt = self.halt_score(context).reshape(1)
            scores = torch.cat([scores, halt])  # poslednji logit = završi liniju
        return scores

    def state_value(self, h):
        return self.value(h.mean(0)).squeeze()
