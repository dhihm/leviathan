"""Example B: Linear(24->96) -> ReLU -> [Linear(96->48) -> Linear(48->96)]
-> ReLU -> Linear(96->10) -> ReLU. Differs from example A: ModuleList loop,
one nn.ReLU module reused at three call sites, two consecutive Linear layers
(linear bottleneck), final ReLU output. Trusted local Python."""

import torch
from torch import nn

INPUT_DIM = 24


class BottleneckStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(INPUT_DIM, 96)
        self.bottleneck = nn.ModuleList([nn.Linear(96, 48), nn.Linear(48, 96)])
        self.head = nn.Linear(96, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.expand(x))
        for layer in self.bottleneck:
            x = layer(x)
        x = self.act(x)
        return self.act(self.head(x))


def build(seed):
    torch.manual_seed(seed)
    return BottleneckStack()
