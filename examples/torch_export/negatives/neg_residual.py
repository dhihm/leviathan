"""NEGATIVE sample: residual connection.

forward returns fc2(fc1(x) + x); the exported graph contains
aten.add.Tensor and a value with two consumers — both rejected (no silent
branch dropping).
"""

import torch
from torch import nn

INPUT_DIM = 16


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x) + x)


def build(seed):
    torch.manual_seed(seed)
    return ResidualNet()
