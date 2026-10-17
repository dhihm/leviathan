"""NEGATIVE sample: data-dependent control flow.

forward branches on x.sum() > 0. torch.export specializes one branch on the
example input; this exporter's differential semantics gate (original module
vs exported graph on 8 seeded probes) must catch the divergence and reject
fail-closed instead of silently keeping one branch.
"""

import torch
from torch import nn

INPUT_DIM = 16


class BranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 8)
        self.fc2 = nn.Linear(INPUT_DIM, 8)
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        if x.sum() > 0:
            x = self.fc1(x)
        else:
            x = self.fc2(torch.relu(x))
        return self.out(x)


def build(seed):
    torch.manual_seed(seed)
    return BranchNet()
