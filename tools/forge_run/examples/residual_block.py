#!/usr/bin/env python3
"""Example DAG model A: 32 -> 96 -> 32 linear block with ReLU and a residual add."""

import torch
from torch import nn

INPUT_DIM = 32


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 96)
        self.fc2 = nn.Linear(96, 32)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h) + x


def build(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return ResidualBlock().eval()
