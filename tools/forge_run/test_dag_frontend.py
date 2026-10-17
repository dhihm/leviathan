'''Tests for dag_frontend using real torch.export captures of nn.Modules.'''
import os
import struct
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_frontend import FORMAT, validate_and_lower_dag  # noqa: E402


def capture(model, dim):
    return torch.export.export(model.eval(), (torch.randn(1, dim),), strict=True)


def interpret(graph, blob, layouts, x):
    '''Independent pure-Python evaluation of the returned DAG and bytes.'''
    f = struct.unpack(f'<{len(blob) // 4}f', blob)
    lay = {L['nodeID']: L for L in layouts}
    vals = {0: list(x)}
    for n in graph['nodes']:
        ins = [vals[i] for i in n['inputs']]
        if n['op'] == 'dense':
            L = lay[n['id']]
            i_, o, w, bo = L['inDim'], L['outDim'], L['weightOffset'], L['biasOffset']
            out = [f[bo + j] + sum(ins[0][k] * f[w + k * o + j] for k in range(i_))
                   for j in range(o)]
        elif n['op'] == 'relu':
            out = [max(v, 0.0) for v in ins[0]]
        else:
            assert len(ins[0]) == len(ins[1])
            out = [a + b for a, b in zip(ins[0], ins[1])]
        assert len(out) == n['outputDim']
        vals[n['id']] = out
    return vals[graph['outputValue']]


def shape(g):
    return [(n['op'], n['inputs'], n['outputDim']) for n in g['nodes']]


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(6, 10), nn.Linear(10, 6)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))


class MultiBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.p, self.q = nn.Linear(5, 8), nn.Linear(5, 8)
        self.r, self.out = nn.Linear(5, 3), nn.Linear(8, 3)

    def forward(self, x):
        return self.out(torch.relu(torch.relu(self.p(x)) + self.q(x))) + self.r(x)


class Alias(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)
        with torch.no_grad():
            self.fc.bias.fill_(-2.0)

    def forward(self, x):
        h = self.fc(x)
        return h + torch.relu(h)


class Two(nn.Module):
    def __init__(self, fn, k=4):
        super().__init__()
        self.a, self.b, self.fn = nn.Linear(4, 4), nn.Linear(4, k), fn

    def forward(self, x):
        return self.fn(self, x)


class DagFrontendTest(unittest.TestCase):
    def check(self, make, dim):
        torch.manual_seed(0)
        model = make().eval()
        g, blob, lay, ops = validate_and_lower_dag(capture(model, dim), dim)
        self.assertEqual((g['format'], g['inputDim']), (FORMAT, dim))
        ids = [n['id'] for n in g['nodes']]
        self.assertEqual(ids, list(range(1, len(ids) + 1)))
        self.assertEqual(g['outputValue'], ids[-1])
        uses = {i: 0 for i in range(len(ids) + 1)}
        for n in g['nodes']:
            self.assertTrue(all(i < n['id'] for i in n['inputs']))
            for i in n['inputs']:
                uses[i] += 1
        self.assertEqual([i for i, c in uses.items() if c == 0], [ids[-1]])
        off = 0
        for L in lay:
            nw = L['inDim'] * L['outDim']
            self.assertEqual((L['weightOffset'], L['weightCount'], L['biasOffset']),
                             (off, nw, off + nw))
            off += nw + L['outDim']
        self.assertEqual(off * 4, len(blob))
        for _ in range(4):
            x = torch.randn(1, dim)
            with torch.no_grad():
                want = model(x)[0].tolist()
            got = interpret(g, blob, lay, x[0].tolist())
            self.assertEqual(len(want), len(got))
            self.assertLess(max(abs(a - b) for a, b in zip(want, got)), 1e-4)
        return g, blob, ops, model

    def test_residual(self):
        g, blob, ops, m = self.check(Residual, 6)
        self.assertEqual(shape(g), [('dense', [0], 10), ('relu', [1], 10),
                                    ('dense', [2], 6), ('add', [0, 3], 6)])
        self.assertEqual(ops, ['aten.linear.default', 'aten.relu.default',
                               'aten.linear.default', 'aten.add.Tensor'])
        self.assertEqual(blob[:240], struct.pack(
            '<60f', *m.fc1.weight.detach().t().reshape(-1).tolist()))
        with self.assertRaisesRegex(ValueError, 'shape'):
            validate_and_lower_dag(capture(m, 6), 7)

    def test_multibranch_projection(self):
        g, _, _, _ = self.check(MultiBranch, 5)
        self.assertEqual(shape(g), [
            ('dense', [0], 8), ('relu', [1], 8), ('dense', [0], 8), ('add', [2, 3], 8),
            ('relu', [4], 8), ('dense', [5], 3), ('dense', [0], 3), ('add', [6, 7], 3)])

    def test_alias_canary_negative_h(self):
        g, _, _, m = self.check(Alias, 4)
        self.assertEqual(shape(g), [('dense', [0], 6), ('relu', [1], 6), ('add', [1, 2], 6)])
        with torch.no_grad():
            self.assertTrue((m.fc(torch.randn(1, 4)) < 0).any())

    def test_rejections(self):
        cases = [
            ('sigmoid', lambda m, x: torch.sigmoid(m.a(x)) + m.b(x), 'aten.sigmoid', {}),
            ('broadcast', lambda m, x: m.a(x) + m.b(x), 'broadcast', {'k': 1}),
            ('alpha2', lambda m, x: torch.add(m.a(x), m.b(x), alpha=2), 'alpha', {}),
            ('scalar', lambda m, x: m.a(x) + m.b(x) + 1.0, 'live tensor value|add.Scalar', {}),
            ('unused', lambda m, x: m.a(x), 'unused', {}),
            ('shared', lambda m, x: m.a(m.a(x)) + m.b(x), 'more than one linear', {}),
            ('nonfinite', lambda m, x: m.a(x) + m.b(x), 'non-finite', {'poison': True}),
        ]
        for name, fn, pattern, opts in cases:
            with self.subTest(name):
                torch.manual_seed(0)
                m = Two(fn, k=opts.get('k', 4))
                if opts.get('poison'):
                    with torch.no_grad():
                        m.a.weight[0, 0] = float('nan')
                try:
                    ep = capture(m, 4)
                except Exception as exc:
                    if opts.get('poison'):
                        self.skipTest(f'export rejects non-finite weights: {exc}')
                    raise
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_and_lower_dag(ep, 4)


if __name__ == '__main__':
    unittest.main()
