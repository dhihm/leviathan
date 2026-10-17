"""NEGATIVE sample: unsupported shape transform.

forward reshapes the activation with aten.view; view is not in the
linear/relu subset, so the exported graph is rejected with the op named
(even though the reshape here preserves the [1, 12] shape).
"""

import torch
from torch import nn

INPUT_DIM = 16


class ReshapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 12)
        self.fc2 = nn.Linear(12, 4)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h.view(1, 12))


def build(seed):
    torch.manual_seed(seed)
    return ReshapeNet()
