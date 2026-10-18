"""A batch over a paged KV cache with scattered, interleaved blocks gives, for every token, the
logits of the single-sequence model run on that sequence alone."""
import unittest

import torch

from ir_export import tiny_config
from ornith_model import OrnithModel, OrnithModelPaged
from test_layers import rope_tables


def randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)


class BlockPool:
    """Hands out physical blocks round robin over sequences so that their blocks interleave."""

    def __init__(self, blocks, block_size):
        self.free = list(range(blocks))
        self.size = block_size
        self.owned = {}

    def ensure(self, seq, pos):
        blocks = self.owned.setdefault(seq, [])
        while len(blocks) * self.size <= pos:
            blocks.append(self.free.pop())  # from the end: ids are not increasing per sequence
        return blocks

    def flat(self, seq, pos):
        return self.ensure(seq, pos)[pos // self.size] * self.size + pos % self.size


class PagedModelTest(unittest.TestCase):
    BS = 4

    def setUp(self):
        self.cfg = tiny_config()
        self.single = OrnithModel(self.cfg).eval()
        randomize(self.single, 21)
        self.paged = OrnithModelPaged(self.cfg).eval()
        self.paged.load_state_dict(self.single.state_dict())

    def single_states(self):
        st = []
        for lay in self.single.state_layout():
            if lay[0] == "linear":
                st += [torch.zeros(*lay[1]), torch.zeros(*lay[2])]
            else:
                st += [torch.zeros(*lay[1]), torch.zeros(*lay[1])]
        return st

    def reference(self, seqs):
        want = {}
        with torch.no_grad():
            for sid, toks in seqs.items():
                st = self.single_states()
                logits = []
                for pos, tok in enumerate(toks):
                    cos, sin = rope_tables(self.cfg, 1, start=pos)
                    res = self.single(torch.tensor([tok]), cos, sin, *st)
                    logits.append(res[0][0])
                    new = list(res[1:])
                    for i, lay in enumerate(self.single.state_layout()):
                        if lay[0] == "full":
                            new[2 * i] = torch.cat([st[2 * i], new[2 * i]])
                            new[2 * i + 1] = torch.cat([st[2 * i + 1], new[2 * i + 1]])
                    st = new
                want[sid] = logits
        return want

    def test_scattered_blocks_equal_single_sequence_runs(self):
        seqs = {0: [5, 17, 3, 42, 8, 30, 11, 4, 9], 1: [9, 1, 47, 30, 2, 6], 2: [20, 21, 7]}
        slots, blocks, max_blocks = 3, 12, 4
        want = self.reference(seqs)
        pool = BlockPool(blocks, self.BS)
        pools = []
        for lay in self.paged.pool_layout(slots, blocks, self.BS):
            if lay[0] == "linear":
                pools += [torch.zeros(*lay[1]), torch.zeros(*lay[2])]
            else:
                pools += [torch.zeros(*lay[1]), torch.zeros(*lay[1])]
        plan = [
            [(0, 0, 5), (1, 0, 2)],
            [(0, 5, 1), (1, 2, 1), (2, 0, 3)],
            [(0, 6, 3), (1, 3, 3)],
        ]
        got = {sid: {} for sid in seqs}
        with torch.no_grad():
            for step in plan:
                ids, slot, pos, mapping, coss, sins, owners = [], [], [], [], [], [], []
                for sid, start, n in step:
                    for j in range(n):
                        p = start + j
                        ids.append(seqs[sid][p])
                        slot.append(sid)
                        pos.append(p)
                        mapping.append(pool.flat(sid, p))
                        cos, sin = rope_tables(self.cfg, 1, start=p)
                        coss.append(cos)
                        sins.append(sin)
                        owners.append((sid, p))
                table = torch.zeros(slots, max_blocks, dtype=torch.long)
                for sid, blks in pool.owned.items():
                    table[sid, : len(blks)] = torch.tensor(blks)
                res = self.paged(torch.tensor(ids), torch.cat(coss), torch.cat(sins), torch.tensor(slot), torch.tensor(pos), torch.tensor(mapping), table, *pools)
                pools = list(res[1:])
                for row, (sid, p) in zip(res[0], owners):
                    got[sid][p] = row
        # The blocks really are scattered: sequence 0 owns non-increasing block ids.
        self.assertNotEqual(pool.owned[0], sorted(pool.owned[0]))
        for sid, toks in seqs.items():
            for p in range(len(toks)):
                torch.testing.assert_close(got[sid][p], want[sid][p], rtol=1e-4, atol=1e-4, msg=f"sequence {sid} position {p}")


if __name__ == "__main__":
    unittest.main()
