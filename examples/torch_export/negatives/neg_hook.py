"""NEGATIVE sample: attached forward hook.

build() registers a forward hook on a child module before returning; the
exporter rejects hooks by module name instead of inlining them into the
graph.
"""

import torch
from torch import nn

INPUT_DIM = 16


class HookNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def build(seed):
    torch.manual_seed(seed)
    model = HookNet()
    model.fc1.register_forward_hook(
        lambda module, args, output: output * 2)
    return model
