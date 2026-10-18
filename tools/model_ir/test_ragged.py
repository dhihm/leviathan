"""A ragged batch of tokens from several sequences gives, for every token, the logits that the
single-sequence model gives for that sequence run on its own."""
import unittest

import torch

from ir_export import tiny_config
from ornith_model import OrnithModel, OrnithModelRagged
from test_layers import rope_tables


def randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)


class RaggedModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_config()
        self.single = OrnithModel(self.cfg).eval()
        randomize(self.single, 21)
        self.ragged = OrnithModelRagged(self.cfg).eval()
        self.ragged.load_state_dict(self.single.state_dict())

    def single_states(self):
        st = []
        for lay in self.single.state_layout():
            if lay[0] == "linear":
                st += [torch.zeros(*lay[1]), torch.zeros(*lay[2])]
            else:
                st += [torch.zeros(*lay[1]), torch.zeros(*lay[1])]
        return st
    def test_mixed_batches_equal_single_sequence_runs(self):
        seqs = {0: [5, 17, 3, 42, 8, 30, 11], 1: [9, 1, 47, 30, 2], 2: [20, 21]}
        slots, max_len = 3, 16
        # Per-sequence reference through the single-sequence model with an explicit cache.
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

        pools = []
        for lay in self.ragged.pool_layout(slots, max_len):
            if lay[0] == "linear":
                pools += [torch.zeros(*lay[1]), torch.zeros(*lay[2])]
            else:
                pools += [torch.zeros(*lay[1]), torch.zeros(*lay[1])]

        # Steps: prefill-like segments of different lengths, then single-token decode steps.
        plan = [
            [(0, 0, 3), (1, 0, 2)],  # (sequence, first position, count)
            [(0, 3, 1), (1, 2, 1), (2, 0, 2)],
            [(0, 4, 3), (1, 3, 2)],
        ]
        got = {sid: {} for sid in seqs}
        with torch.no_grad():
            for step in plan:
                ids, slot, pos, coss, sins, owners = [], [], [], [], [], []
                for sid, start, n in step:
                    for j in range(n):
                        p = start + j
                        ids.append(seqs[sid][p])
                        slot.append(sid)
                        pos.append(p)
                        cos, sin = rope_tables(self.cfg, 1, start=p)
                        coss.append(cos)
                        sins.append(sin)
                        owners.append((sid, p))
                res = self.ragged(torch.tensor(ids), torch.cat(coss), torch.cat(sins), torch.tensor(slot), torch.tensor(pos), *pools)
                pools = list(res[1:])
                for row, (sid, p) in zip(res[0], owners):
                    got[sid][p] = row
        for sid, toks in seqs.items():
            for p in range(len(toks)):
                torch.testing.assert_close(got[sid][p], want[sid][p], rtol=1e-4, atol=1e-4, msg=f"sequence {sid} position {p}")


if __name__ == "__main__":
    unittest.main()
