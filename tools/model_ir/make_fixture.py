"""Writes a test fixture for the Rust IR interpreter.

    python make_fixture.py <out_dir>

A small random Ornith-shaped model (4 layers: three linear attention, one full attention) is
exported to the IR, and the same weights, inputs (with non-empty states and KV cache) and the
expected outputs of the Python model are written in the tensor pack format. Before writing, the
functional model is checked against the Hugging Face Qwen3_5ForCausalLM on a whole sequence.

Tensor pack ("LEVT"): magic, u32 count, then per tensor: u16 name length, name (utf-8), u8 dtype
(0 float32, 1 int64), u8 rank, rank x u64 dims, u64 byte length, raw little-endian data.
"""
import os
import struct
import sys

import numpy as np
import torch

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from ir_export import build_and_export, serialize, tiny_config
from ornith_model import OrnithModel
from test_layers import rope_tables

import json

DTYPES = {torch.float32: 0, torch.int64: 1}


def write_pack(path, tensors):
    with open(path, "wb") as f:
        f.write(b"LEVT")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors:
            t = t.detach().contiguous()
            nb = name.encode("utf-8")
            f.write(struct.pack("<H", len(nb)) + nb)
            f.write(struct.pack("<BB", DTYPES[t.dtype], t.dim()))
            for d in t.shape:
                f.write(struct.pack("<Q", d))
            raw = t.numpy().tobytes()
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)


def randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    cfg = tiny_config()
    model = OrnithModel(cfg).eval()
    randomize(model, 7)

    # The functional model equals the Hugging Face model on a whole sequence.
    hf = Qwen3_5ForCausalLM(cfg).eval()
    hf.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([5, 17, 3, 42, 8, 30, 11])
    cos, sin = rope_tables(cfg, ids.shape[0])
    states = []
    for lay in model.state_layout():
        if lay[0] == "linear":
            states += [torch.zeros(*lay[1]), torch.zeros(*lay[2])]
        else:
            states += [torch.zeros(*lay[1]), torch.zeros(*lay[1])]
    with torch.no_grad():
        want = hf(input_ids=ids[None]).logits[0]
        got = model(ids, cos, sin, *states)[0]
    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)
    print("functional model == Hugging Face Qwen3_5ForCausalLM (max abs diff %.2e)" % (got - want).abs().max())

    # Fixture inputs: new tokens after a prefix, with non-empty states and cache.
    g = torch.Generator().manual_seed(99)
    t, past = 5, 3
    ids = torch.tensor([9, 1, 47, 30, 2])
    cos, sin = rope_tables(cfg, t, start=past)
    states = []
    for lay in model.state_layout():
        if lay[0] == "linear":
            states += [torch.randn(*lay[1], generator=g) * 0.5, torch.randn(*lay[2], generator=g) * 0.1]
        else:
            shp = (past,) + tuple(lay[1][1:])
            states += [torch.randn(*shp, generator=g), torch.randn(*shp, generator=g)]
    with torch.no_grad():
        outs = model(ids, cos, sin, *states)

    _, ep = build_and_export(cfg, meta=False)
    ir = serialize(ep, cfg)
    with open(os.path.join(out, "ir.json"), "w", encoding="utf-8") as f:
        json.dump(ir, f)
    write_pack(os.path.join(out, "weights.pack"), list(model.state_dict().items()))
    inputs = [("token_ids", ids), ("cos", cos), ("sin", sin)] + [(f"states_{i}", s) for i, s in enumerate(states)]
    write_pack(os.path.join(out, "inputs.pack"), inputs)
    write_pack(os.path.join(out, "expected.pack"), [(f"out_{i}", o) for i, o in enumerate(outs)])
    print(f"wrote fixture to {out}: {len(ir['nodes'])} nodes, {len(ir['params'])} params, {len(outs)} outputs")


if __name__ == "__main__":
    main()
