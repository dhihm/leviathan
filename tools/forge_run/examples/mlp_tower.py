"""Example A: Linear(128->512) -> F.relu -> Linear(512->256) -> torch.relu
-> Linear(256->64). Custom forward (not nn.Sequential), no output
activation. Executed as trusted local Python by leviathan_forge.py."""

import torch
import torch.nn.functional as F
from torch import nn

INPUT_DIM = 128


class MlpTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 512)
        self.fc2 = nn.Linear(512, 256)
        self.head = nn.Linear(256, 64)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        hidden = torch.relu(self.fc2(hidden))
        return self.head(hidden)


def build(seed):
    torch.manual_seed(seed)
    return MlpTower()
