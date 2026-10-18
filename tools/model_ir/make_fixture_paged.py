"""Test fixture for the paged-KV IR (4-token blocks, scattered block ids): same formats as make_fixture.py.

    python make_fixture_paged.py <out_dir>
"""
import json
import os
import sys

import torch

from ir_export import build_and_export_paged, serialize, tiny_config
from make_fixture import randomize, write_pack
from ornith_model import OrnithModelPaged
from test_layers import rope_tables

BS = 4


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    cfg = tiny_config()
    model = OrnithModelPaged(cfg).eval()
    randomize(model, 7)
    slots, blocks, max_blocks = 3, 12, 4
    # (slot, first position, tokens); the blocks of every sequence are scattered over the pool.
    batch = [(0, 3, [9, 1, 47]), (1, 2, [30]), (2, 0, [2, 5])]
    owned = {0: [9, 4], 1: [11], 2: [2]}
    ids, slot, pos, mapping, coss, sins = [], [], [], [], [], []
    for s, start, toks in batch:
        for j, tok in enumerate(toks):
            p = start + j
            ids.append(tok)
            slot.append(s)
            pos.append(p)
            mapping.append(owned[s][p // BS] * BS + p % BS)
            cos, sin = rope_tables(cfg, 1, start=p)
            coss.append(cos)
            sins.append(sin)
    table = torch.zeros(slots, max_blocks, dtype=torch.long)
    for s, b in owned.items():
        table[s, : len(b)] = torch.tensor(b)
    g = torch.Generator().manual_seed(123)
    pools = []
    for lay in model.pool_layout(slots, blocks, BS):
        if lay[0] == "linear":
            pools += [torch.randn(*lay[1], generator=g) * 0.5, torch.randn(*lay[2], generator=g) * 0.1]
        else:
            pools += [torch.randn(*lay[1], generator=g), torch.randn(*lay[1], generator=g)]
    inputs = [
        ("token_ids", torch.tensor(ids)),
        ("cos", torch.cat(coss)),
        ("sin", torch.cat(sins)),
        ("slot", torch.tensor(slot)),
        ("pos", torch.tensor(pos)),
        ("slot_mapping", torch.tensor(mapping)),
        ("block_table", table),
    ] + [(f"pools_{i}", p) for i, p in enumerate(pools)]
    with torch.no_grad():
        outs = model(*[t for _, t in inputs])
    _, ep = build_and_export_paged(cfg, meta=False, block_size=BS)
    ir = serialize(ep, cfg)
    ir["layout"] = "paged-v1"
    with open(os.path.join(out, "ir.json"), "w", encoding="utf-8") as f:
        json.dump(ir, f)
    write_pack(os.path.join(out, "weights.pack"), list(model.state_dict().items()))
    write_pack(os.path.join(out, "inputs.pack"), inputs)
    write_pack(os.path.join(out, "expected.pack"), [(f"out_{i}", o) for i, o in enumerate(outs)])
    print(f"wrote paged fixture to {out}: {len(ir['nodes'])} nodes, {len(ir['inputs'])} inputs, {len(outs)} outputs")


if __name__ == "__main__":
    main()