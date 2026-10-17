#!/usr/bin/env python3
"""Example DAG model B: 24 -> 48 trunk with ReLU, a projected skip (24 -> 48),
a second linear branch (48 -> 48) and a ReLU branch, all added elementwise."""

import torch
from torch import nn

INPUT_DIM = 24


class ProjectedBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(24, 48)
        self.branch = nn.Linear(48, 48)
        self.skip = nn.Linear(24, 48)
        self.second = nn.Linear(48, 48)

    def forward(self, x):
        h = torch.relu(self.trunk(x))
        out = self.branch(h) + self.skip(x)
        return out + torch.relu(self.second(h))


def build(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return ProjectedBranches().eval()
