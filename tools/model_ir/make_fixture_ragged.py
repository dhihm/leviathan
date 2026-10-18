"""Test fixture for the ragged (multi-sequence) IR: same file formats as make_fixture.py.

    python make_fixture_ragged.py <out_dir>

One batch of six tokens from three sequences (slots 0, 1, 2 at different positions, 3 + 1 + 2
tokens) over pools that already hold random state, so reads of earlier positions and of earlier
recurrent state are exercised.
"""
import json
import os
import sys

import torch

from ir_export import build_and_export_ragged, serialize, tiny_config
from make_fixture import randomize, write_pack
from ornith_model import OrnithModelRagged
from test_layers import rope_tables


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    cfg = tiny_config()
    model = OrnithModelRagged(cfg).eval()
    randomize(model, 7)

    slots, max_len = 3, 8
    batch = [(0, 3, [9, 1, 47]), (1, 2, [30]), (2, 0, [2, 5])]  # (slot, first position, tokens)
    ids, slot, pos, coss, sins = [], [], [], [], []
    for s, start, toks in batch:
        for j, tok in enumerate(toks):
            ids.append(tok)
            slot.append(s)
            pos.append(start + j)
            cos, sin = rope_tables(cfg, 1, start=start + j)
            coss.append(cos)
            sins.append(sin)
    g = torch.Generator().manual_seed(123)
    pools = []
    for lay in model.pool_layout(slots, max_len):
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
    ] + [(f"pools_{i}", p) for i, p in enumerate(pools)]
    with torch.no_grad():
        outs = model(*[t for _, t in inputs])

    _, ep = build_and_export_ragged(cfg, meta=False)
    ir = serialize(ep, cfg)
    ir["layout"] = "ragged-v1"
    with open(os.path.join(out, "ir.json"), "w", encoding="utf-8") as f:
        json.dump(ir, f)
    write_pack(os.path.join(out, "weights.pack"), list(model.state_dict().items()))
    write_pack(os.path.join(out, "inputs.pack"), inputs)
    write_pack(os.path.join(out, "expected.pack"), [(f"out_{i}", o) for i, o in enumerate(outs)])
    print(f"wrote ragged fixture to {out}: {len(ir['nodes'])} nodes, {len(ir['inputs'])} inputs, {len(outs)} outputs")


if __name__ == "__main__":
    main()
