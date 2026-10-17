"""NEGATIVE sample: shared parameter.

fc2 reuses fc1's Linear module; the same parameter would appear at two
positions in the weight blob. The exporter detects the shared object via
deduplication-free traversal and rejects it by name.
"""

import torch
from torch import nn

INPUT_DIM = 16


class SharedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 16)
        self.fc2 = self.fc1
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def build(seed):
    torch.manual_seed(seed)
    return SharedNet()
