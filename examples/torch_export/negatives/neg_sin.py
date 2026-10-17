"""NEGATIVE sample: unsupported op torch.sin.

The exported graph contains aten.sin.default, which is outside the
linear/relu subset and must be rejected with the op named.
"""

import torch
from torch import nn

INPUT_DIM = 16


class SinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(torch.sin(self.fc1(x)))


def build(seed):
    torch.manual_seed(seed)
    return SinNet()
