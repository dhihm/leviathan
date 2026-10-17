"""Example PyTorch model definition for the LeviathanForge exporter.

build(seed) constructs a small float32 MLP with reproducible random
initialization: different seeds produce genuinely different parameter
weights (and therefore different reference outputs), which the exporter's
manifest records for parity checking in Swift. These are random-init
parameters, not a trained model.
"""

import torch
from torch import nn


def build(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
    )
