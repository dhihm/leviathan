"""Sample model for the strict torch.export frontend
(tools/torch_frontend/export_torch_graph.py).

This is deliberately NOT a plain nn.Sequential: forward() iterates a
ModuleList, mixes functional F.linear/torch.relu with module calls, and the
resulting ATen graph therefore differs from the child-module labels. The
exporter captures semantics from the exported graph, not from module
structure, and validates the pure Linear/ReLU chain.
"""

import torch
from torch import nn
import torch.nn.functional as F

INPUT_DIM = 64


class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.relu(self.proj(x))


class CustomForwardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(INPUT_DIM, 96)
        self.blocks = nn.ModuleList([Block(96, 96), Block(96, 32)])
        self.head = nn.Linear(32, 8)
        # A child module forward() never calls: the captured graph is built
        # from executed semantics, so this Linear appears in no dense op.
        self.unused = nn.Linear(96, 96)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        for block in self.blocks:
            x = block(x)
        x = F.relu(x)
        return F.linear(x, self.head.weight, self.head.bias)


def build(seed):
    generator = torch.Generator().manual_seed(seed)
    model = CustomForwardMLP()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.rand(param.shape, generator=generator) * 0.2 - 0.1)
    return model
