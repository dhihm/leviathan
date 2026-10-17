#!/usr/bin/env python3
"""Bounded stdlib-only tests for the MoE region capture frontend.

Runs export_moe_region.py as a subprocess over positive and negative
region models (tiny 64/32/4 geometry plus the real 2816/704/8 capture) and
verifies: positives capture with graph digest + independently re-derived
semantics + torch bit-parity of the exported module against the original
forward and an independent hand loop; negatives fail closed naming the
offense and leave NO output files.

NOT a general-compiler claim: the supported region is exactly the routed
top-K slot dataflow (no-bias linear / split / GELU(tanh) / mul / stack).
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
CAPTURE = os.path.join(HERE, "export_moe_region.py")
NEGDIR = os.path.join(HERE, "negatives")

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"PASS: {name}")
    else:
        FAIL += 1
        print(f"FAIL: {name} {detail}")


def run_capture(model, out, config=None):
    cmd = [PYTHON, CAPTURE, model, out] + ([config] if config else [])
    return subprocess.run(cmd, capture_output=True, text=True)


def expect_reject(name, model_body, needle):
    with tempfile.TemporaryDirectory() as td:
        model = os.path.join(td, "neg.py")
        out = os.path.join(td, "neg.json")
        source = NEG_TEMPLATE.replace("{{BODY}}", textwrap.dedent(model_body))
        # A syntax-broken negative proves nothing: compile it first.
        compile(source, model, "exec")
        with open(model, "w") as f:
            f.write(source)
        r = run_capture(model, out)
        combined = r.stdout + r.stderr
        check(f"neg {name}: nonzero exit", r.returncode != 0,
              f"rc={r.returncode}")
        check(f"neg {name}: names the offense ({needle})", needle in combined,
              combined.strip()[-300:])
        check(f"neg {name}: no output files", not os.path.exists(out))


NEG_TEMPLATE = """import torch
import torch.nn as nn
import torch.nn.functional as F
{{BODY}}
"""


def main():
    os.makedirs(NEGDIR, exist_ok=True)

    # ---- positive: real Gemma-4 26B geometry (meta tensors only).
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "region.json")
        r = run_capture(os.path.join(HERE, "model_moe_region.py"), out)
        check("pos gemma dims: exit 0", r.returncode == 0, r.stderr[-400:])
        if r.returncode == 0:
            region = json.load(open(out))
            check("pos gemma dims: derived dims",
                  (region["hidden"], region["expertIntermediate"],
                   region["topK"], region["gateUpOut"]) == (2816, 704, 8, 1408),
                  str(region))
            g = region["graph"]
            check("pos graph: node count", len(g["nodes"]) == 3 + 8 * 9 + 2,
                  str(len(g["nodes"])))
            check("pos graph: census ops",
                  set(region["torch"]["atenOps"]) == {
                      "aten.gelu.default", "aten.linear.default",
                      "aten.mul.Tensor", "aten.select.int",
                      "aten.split.Tensor", "aten.stack.default",
                      "operator.getitem"},
                  str(region["torch"]["atenOps"]))
            check("pos advisory: torch block advisory-only",
                  region["torch"]["advisoryOnly"] is True)
            check("pos slotOps derived",
                  region["slotOps"] == [
                      "q4_0.expert-slice-linear-gate-up",
                      "gelu-tanh-split-mul",
                      "q4_0.expert-slice-linear-down"], str(region["slotOps"]))
            # Independent digest recomputation from the emitted graph.
            sys.path.insert(0, HERE)
            spec = importlib.util.spec_from_file_location(
                "exp", CAPTURE)
            exp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exp)
            lines = []
            for n in g["nodes"]:
                meta = n["meta"]
                meta_c = "n" if meta is None else exp.canon(meta)
                lines.append("|".join([n["name"], n["op"], n["target"],
                                       exp.canon(n["args"]),
                                       exp.canon(n["kwargs"]), meta_c]))
            lines.append("inputs|" + exp.canon(g["inputs"]))
            lines.append("outputs|" + exp.canon(g["outputs"]))
            import hashlib
            digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
            check("pos graph digest matches independent recomputation",
                  digest == region["graphSha256"])
            # Same-seed determinism: the normalized graph is stable.
            out2 = os.path.join(td, "region2.json")
            r2 = run_capture(os.path.join(HERE, "model_moe_region.py"), out2)
            check("pos deterministic graph", r2.returncode == 0 and
                  json.load(open(out2))["graph"] == g)

    # ---- positive: torch bit-parity on a tiny REAL-tensor region.
    sys.path.insert(0, HERE)
    from model_moe_region import build
    torch = importlib.import_module("torch")
    F = torch.nn.functional
    torch.manual_seed(11)
    m = build({"hidden": 64, "inter": 32, "topK": 4})
    x = torch.randn(1, 64)
    gu = torch.randn(4, 64, 64)
    dn = torch.randn(4, 64, 32)
    ep = torch.export.export(m, (x, gu, dn), strict=True)
    got = ep.module()(x, gu, dn)
    want = m(x, gu, dn)
    check("pos export parity: exported module == original forward",
          got.shape == want.shape and torch.equal(got, want))
    ref = torch.stack([F.linear(
        (lambda g_, u_: F.gelu(g_, approximate="tanh") * u_)(
            *F.linear(x, gu[s]).split(32, dim=-1)),
        dn[s]) for s in range(4)], dim=0)
    check("pos export parity: == independent hand loop",
          torch.equal(got, ref))
    check("pos export parity: finite", bool(got.isfinite().all()))

    # ---- negatives: real nn.Modules with altered semantics.
    negs = [
        ("erf-gelu",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[slot], bias=None)
            gate,up=gu.split(self.inter,dim=-1)
            act=F.gelu(gate)*up
            outs.append(F.linear(act, down_weights[slot], bias=None))
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "approximate"),
        ("slot-reverse",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[self.top_k-1-slot], bias=None)
            gate,up=gu.split(self.inter,dim=-1)
            act=F.gelu(gate,approximate="tanh")*up
            outs.append(F.linear(act, down_weights[self.top_k-1-slot], bias=None))
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "slot"),
        ("bias",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
        self.b = nn.Parameter(torch.zeros(64, device="meta"))
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[slot], bias=self.b)
            gate,up=gu.split(self.inter,dim=-1)
            act=F.gelu(gate,approximate="tanh")*up
            outs.append(F.linear(act, down_weights[slot], bias=None))
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "USER_INPUT"),
        ("extra-op",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[slot], bias=None)
            gate,up=gu.split(self.inter,dim=-1)
            act=F.gelu(gate,approximate="tanh")*up
            y=F.linear(act, down_weights[slot], bias=None)
            outs.append(y + torch.sin(y) * 0)
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "aten.add.Tensor"),
        ("stack-dim1",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[slot], bias=None)
            gate,up=gu.split(self.inter,dim=-1)
            act=F.gelu(gate,approximate="tanh")*up
            outs.append(F.linear(act, down_weights[slot], bias=None).squeeze(0))
        return torch.stack(outs,dim=1)
def build(config=None): return R()
''', "dim=0"),
        ("mul-swapped",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[slot], bias=None)
            gate,up=gu.split(self.inter,dim=-1)
            act=up*F.gelu(gate,approximate="tanh")
            outs.append(F.linear(act, down_weights[slot], bias=None))
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "REJECT"),
        ("gate-up-indices",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[slot], bias=None)
            up,gate=gu.split(self.inter,dim=-1)
            act=F.gelu(gate,approximate="tanh")*up
            outs.append(F.linear(act, down_weights[slot], bias=None))
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "getitem"),
        ("down-input-reshape",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[slot], bias=None)
            gate,up=gu.split(self.inter,dim=-1)
            act=F.gelu(gate,approximate="tanh")*up
            outs.append(F.linear(act.view(1, 32), down_weights[slot], bias=None))
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "aten.view.default"),
        ("mutation",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        x = x.clone()
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x, gu_weights[slot], bias=None)
            gate,up=gu.split(self.inter,dim=-1)
            act=F.gelu(gate,approximate="tanh")*up
            x += act.mean()
            outs.append(F.linear(act, down_weights[slot], bias=None))
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "REJECT"),
        ("unshared-x",
         '''
class R(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = 32
        self.top_k = 4
        self.hidden = 64
    def forward(self, x, gu_weights, down_weights):
        outs=[]
        for slot in range(self.top_k):
            gu=F.linear(x*1.0 if slot==0 else x, gu_weights[slot], bias=None)
            gate,up=gu.split(self.inter,dim=-1)
            act=F.gelu(gate,approximate="tanh")*up
            outs.append(F.linear(act, down_weights[slot], bias=None))
        return torch.stack(outs,dim=0)
def build(config=None): return R()
''', "REJECT"),
    ]
    for name, body, needle in negs:
        expect_reject(name, body, needle)

    print(f"\n{PASS} PASS, {FAIL} FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
