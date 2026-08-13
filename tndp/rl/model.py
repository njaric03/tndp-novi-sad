# GATv2 encoder + pointer glava za izbor cvora + halt glava + value glava

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from tndp.rl.env import HALT
from tndp.rl.features import num_features, spec

class TndpPolicy(nn.Module):
    def __init__(self, hidden=64, layers=3, heads=4, features="v1"):
        super().__init__()
        # verzija stoji na modelu da je node_features cita sa politike, umesto da se prosledjuje
        self.features = spec(features)
        self.embed = nn.Linear(num_features(features), hidden)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=2)
            for _ in range(layers)])
        # ulaz: [embedding cvora, globalni kontekst, embedding kraja na koji se kaci]
        self.node_score = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.halt_score = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x, edge_index, edge_attr):
        h = self.embed(x)
        for conv in self.convs:
            h = h + torch.relu(conv(h, edge_index, edge_attr))
        return h

    # logiti nad akcijama: 2n parova (kraj, cvor) po rasporedu maske [HEAD nad svim cvorovima, TAIL nad svim cvorovima]
    def action_logits(self, h, decision, mask, ends):
        n = h.shape[0]
        context = h.mean(0)
        ctx = context.expand(n, -1)
        if ends is None:  # start linije: nema kraja, kotva je globalni kontekst
            anchors = (ctx, ctx)
        else:
            anchors = (h[ends[0]].expand(n, -1), h[ends[1]].expand(n, -1))
        scores = torch.cat([
            self.node_score(torch.cat([h, ctx, a], dim=1)).squeeze(-1)
            for a in anchors])
        scores = scores.masked_fill(
            ~torch.as_tensor(mask, dtype=torch.bool).reshape(-1), -torch.inf)
        if decision == HALT:
            halt = self.halt_score(context).reshape(1)
            scores = torch.cat([scores, halt])  # poslednji logit = zavrsi liniju
        return scores

    def state_value(self, h):
        return self.value(h.mean(0)).squeeze()
