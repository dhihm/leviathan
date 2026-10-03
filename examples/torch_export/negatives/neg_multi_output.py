"""NEGATIVE sample: multiple outputs.

forward returns a tuple (logits, hidden); the exporter requires a single
output tensor.
"""

import torch
from torch import nn

INPUT_DIM = 16


class MultiOutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h), h


def build(seed):
    torch.manual_seed(seed)
    return MultiOutNet()
