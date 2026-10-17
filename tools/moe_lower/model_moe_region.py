"""Routed MoE expert region model for LeviathanForge region lowering.

This is a REAL PyTorch nn.Module whose forward expresses the semantics of
ONE bounded region of the Gemma-4 26B-A4B model: the routed top-K expert
slots between `pre_ffw_norm_2` and the ordered weighted merge. The runtime
host router (outside this region) selects the expert ids and merge weights;
this region computes, per ordered slot, the pure dataflow

    gu   = linear(x, GU[slot], bias=None)          # [1, 2*inter] gate|up
    gate, up = split(gu, inter, dim=-1)
    act  = gelu(gate, approximate="tanh") * up     # [1, inter]
    y    = linear(act, DN[slot], bias=None)        # [1, hidden]

and returns the slot outputs stacked in exact slot order [topK, 1, hidden].

`build(config)` returns the module; the capture script exports it with
torch.export(strict=True) on META tensors (static shapes only, no weight
allocation). This is a REGION-level input — not a claim of arbitrary
whole-model Python compilation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERoutedRegion(nn.Module):
    def __init__(self, hidden, inter, top_k):
        super().__init__()
        if hidden <= 0 or inter <= 0 or top_k <= 0:
            raise ValueError("hidden/inter/top_k must be positive")
        if hidden % 32 != 0 or inter % 32 != 0:
            raise ValueError("hidden and inter must be multiples of 32 "
                             "(Q4_0 block values)")
        self.hidden = hidden
        self.inter = inter
        self.top_k = top_k

    def forward(self, x, gu_weights, down_weights):
        outs = []
        for slot in range(self.top_k):
            gu = F.linear(x, gu_weights[slot], bias=None)
            gate, up = gu.split(self.inter, dim=-1)
            act = F.gelu(gate, approximate="tanh") * up
            outs.append(F.linear(act, down_weights[slot], bias=None))
        return torch.stack(outs, dim=0)


# The actual Gemma-4 26B-A4B Q4_0 routed region geometry (verified against
# the authorized GGUF by the Swift side's fail-closed crosscheck — this
# default is only the capture-time declaration).
GEMMA4_26B_CONFIG = {"hidden": 2816, "inter": 704, "topK": 8}


def build(config=None):
    cfg = dict(config or GEMMA4_26B_CONFIG)
    return MoERoutedRegion(cfg["hidden"], cfg["inter"], cfg["topK"])
