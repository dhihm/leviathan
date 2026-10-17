"""POSITIVE sample ending with a ReLU (not a dense op).

A custom forward over a ModuleDict whose final op is a ReLU, exercising the
exporter's acceptance of relu-terminated chains.
"""

import torch
from torch import nn
import torch.nn.functional as F

INPUT_DIM = 32


class EndReLUMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({
            "l1": nn.Linear(INPUT_DIM, 24),
            "l2": nn.Linear(24, 12),
        })

    def forward(self, x):
        x = F.relu(self.layers["l1"](x))
        x = self.layers["l2"](x)
        return F.relu(x)


def build(seed):
    generator = torch.Generator().manual_seed(seed)
    model = EndReLUMLP()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.rand(param.shape, generator=generator) * 0.2 - 0.1)
    return model
