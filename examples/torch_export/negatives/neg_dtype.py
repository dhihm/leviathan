"""NEGATIVE sample: non-float32 dtype.

The Linear parameters are float16; the exporter requires float32
parameters/inputs.
"""

import torch
from torch import nn

INPUT_DIM = 16


class HalfNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 8).to(torch.float16)
        self.fc2 = nn.Linear(8, 4).to(torch.float16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.float().to(torch.float16)))).float()


def build(seed):
    torch.manual_seed(seed)
    return HalfNet()
