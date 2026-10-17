#!/usr/bin/env python3
'''Generic static value-DAG frontend for LeviathanForge.

validate_and_lower_dag(ep, input_dim) checks a torch.export(strict=True)
ExportedProgram against a small static subset and lowers it to a
leviathan-value-dag-1 graph plus the ACTUAL exported parameter bytes:

- one user input [1, input_dim] CPU float32, exported PARAMETER placeholders
  only (buffers/constants/tokens/mutations rejected), one user output;
- aten.linear.default(x, weight_param, bias_param): exactly 3 args, no
  kwargs, each parameter consumed by exactly one linear operand slot;
- aten.relu.default(x): exactly 1 arg, no kwargs;
- aten.add.Tensor(a, b): two live tensor values of identical [1, D] shape
  (no broadcasting, no scalars); an alpha kwarg must be numeric 1.

Value IDs are immutable: 0 is the user input, 1..N are call nodes in
topological graph order. Activation fanout is allowed; parameter fanout,
unused parameters and ops that do not reach the output are rejected. Every
violation raises DagFrontendError (a ValueError) naming the offending node.
Weights are never synthesized. Module side conditions (hooks,
parametrizations) remain the caller's responsibility.
'''
import struct

import torch
from torch.export.graph_signature import InputKind, OutputKind

FORMAT = 'leviathan-value-dag-1'
MAX_CALLS = 64
MAX_DIM = 4096
MAX_WEIGHT_BYTES = 16 << 20
LINEAR, RELU, ADD = 'aten.linear.default', 'aten.relu.default', 'aten.add.Tensor'


class DagFrontendError(ValueError):
    pass


def fail(message):
    raise DagFrontendError(f'dag_frontend: {message}')


def expect_static_tensor(node, shape):
    '''Node metadata must be a static CPU float32 tensor of exact shape.'''
    val = node.meta.get('val')
    if not isinstance(val, torch.Tensor):
        fail(f'node {node.name!r} has no tensor metadata (non-tensor value)')
    dims = tuple(val.shape)
    if any(type(d) is not int for d in dims):
        fail(f'node {node.name!r} has non-static shape {dims}')
    if dims != tuple(shape):
        fail(f'node {node.name!r} shape {dims} != required {tuple(shape)}')
    if val.dtype is not torch.float32 or val.device.type != 'cpu':
        fail(f'node {node.name!r} must be CPU float32, got {val.dtype}/{val.device}')


def validate_and_lower_dag(ep, input_dim):
    '''Return (graph, weight_bytes, dense_layouts, aten_ops).'''
    if type(input_dim) is not int or not 0 < input_dim <= MAX_DIM:
        fail(f'input_dim {input_dim!r} must be an int in 1...{MAX_DIM}')
    sig = ep.graph_signature
    outs = sig.output_specs
    if len(outs) != 1 or outs[0].kind is not OutputKind.USER_OUTPUT:
        fail(f'exactly one user output is required (mutations unsupported), '
             f'got {[str(s.kind) for s in outs]}')
    param_target, user = {}, []
    for spec in sig.input_specs:
        if spec.kind is InputKind.PARAMETER:
            param_target[spec.arg.name] = spec.target
        elif spec.kind is InputKind.USER_INPUT:
            user.append(spec.arg.name)
        else:
            fail(f'unsupported input kind {spec.kind} ({spec.arg.name!r} -> {spec.target})')
    if len(user) != 1:
        fail(f'exactly one user input is required, got {len(user)}')
    if getattr(ep, 'constants', None):
        fail(f'exported constants are not supported: {sorted(ep.constants)}')
    nodes = list(ep.graph_module.graph.nodes)
    for n in nodes:
        if n.op not in ('placeholder', 'call_function', 'output'):
            fail(f'unsupported graph construct {n.op} at {n.name!r}')
    holders = {n.name: n for n in nodes if n.op == 'placeholder'}
    if set(holders) != set(param_target) | set(user) or user[0] in param_target:
        fail(f'placeholder/signature mismatch: {sorted(holders)}')
    outputs = [n for n in nodes if n.op == 'output']
    if len(outputs) != 1 or len(outputs[0].args) != 1:
        fail('exactly one output node is required')
    out_vals = outputs[0].args[0]
    if (not isinstance(out_vals, (tuple, list)) or len(out_vals) != 1
            or not isinstance(out_vals[0], torch.fx.Node)):
        fail('the model must return exactly one tensor')
    out_node = out_vals[0]
    calls = [n for n in nodes if n.op == 'call_function']
    if len(calls) > MAX_CALLS:
        fail(f'call count {len(calls)} exceeds {MAX_CALLS}')
    state = ep.state_dict
    x_node = holders[user[0]]
    expect_static_tensor(x_node, (1, input_dim))
    ids, dims = {x_node: 0}, {x_node: input_dim}
    dag, pending, aten_ops, used = [], [], [], {}
    total_floats = 0

    def value(node, arg, k):
        if not isinstance(arg, torch.fx.Node) or arg not in ids:
            shown = arg.name if isinstance(arg, torch.fx.Node) else arg
            fail(f'operand {k} of {node.name!r} must be a live tensor value '
                 f'(user input or earlier op), got {shown!r}')
        return arg

    def param(node, arg, label):
        if not isinstance(arg, torch.fx.Node) or arg.name not in param_target:
            fail(f'linear {label} of {node.name!r} must be an exported parameter placeholder')
        if arg.name in used:
            fail(f'parameter {arg.name!r} is used by more than one linear '
                 f'({used[arg.name]!r} and {node.name!r})')
        used[arg.name] = node.name
        t = state.get(param_target[arg.name])
        if (not isinstance(t, torch.Tensor) or t.dtype is not torch.float32
                or t.device.type != 'cpu'):
            fail(f'linear {label} {arg.name!r} of {node.name!r} must be a CPU float32 tensor')
        expect_static_tensor(arg, tuple(t.shape))
        if not bool(torch.isfinite(t).all()):
            fail(f'linear {label} {arg.name!r} of {node.name!r} is non-finite')
        return t.detach()

    for nid, node in enumerate(calls, 1):
        op = str(node.target)
        aten_ops.append(op)
        if op == LINEAR:
            if node.kwargs or len(node.args) != 3:
                fail(f'linear {node.name!r} must be exactly (input, weight, bias) without kwargs')
            x = value(node, node.args[0], 0)
            w = param(node, node.args[1], 'weight')
            b = param(node, node.args[2], 'bias')
            if w.dim() != 2 or b.dim() != 1:
                fail(f'linear {node.name!r}: weight must be rank 2 and bias rank 1')
            out_dim, in_dim = (int(d) for d in w.shape)
            if in_dim != dims[x] or tuple(b.shape) != (out_dim,) or not 0 < out_dim <= MAX_DIM:
                fail(f'linear {node.name!r}: weight {tuple(w.shape)} / bias '
                     f'{tuple(b.shape)} do not match input dim {dims[x]}')
            total_floats += w.numel() + b.numel()
            if total_floats * 4 > MAX_WEIGHT_BYTES:
                fail(f'weights exceed {MAX_WEIGHT_BYTES} bytes at {node.name!r}')
            pending.append((nid, in_dim, out_dim, w, b))
            entry = ('dense', [ids[x]], out_dim)
        elif op == RELU:
            if node.kwargs or len(node.args) != 1:
                fail(f'relu {node.name!r} must take exactly one arg without kwargs')
            x = value(node, node.args[0], 0)
            entry = ('relu', [ids[x]], dims[x])
        elif op == ADD:
            if len(node.args) != 2 or set(node.kwargs) - {'alpha'}:
                fail(f'add {node.name!r} must be (a, b) with at most alpha=1')
            if 'alpha' in node.kwargs:
                alpha = node.kwargs['alpha']
                if type(alpha) not in (int, float) or alpha != 1:
                    fail(f'add {node.name!r} has unsupported alpha {alpha!r}')
            a = value(node, node.args[0], 0)
            c = value(node, node.args[1], 1)
            if dims[a] != dims[c]:
                fail(f'add {node.name!r} broadcasting is not supported '
                     f'({dims[a]} vs {dims[c]} features)')
            entry = ('add', [ids[a], ids[c]], dims[a])
        else:
            fail(f'unsupported ATen op {op!r} at node {node.name!r}')
        expect_static_tensor(node, (1, entry[2]))
        ids[node], dims[node] = nid, entry[2]
        dag.append({'id': nid, 'op': entry[0], 'inputs': entry[1], 'outputDim': entry[2]})

    if not pending:
        fail('at least one dense (aten.linear.default) op is required')
    if len(set(param_target.values())) != len(param_target):
        fail(f'parameter placeholders alias duplicate state targets: '
             f'{sorted(param_target.values())}')
    used_targets = {param_target[name] for name in used}
    state_keys = set(state)
    extra = sorted(state_keys - used_targets)
    missing = sorted(used_targets - state_keys)
    if extra:
        fail(f'unused exported parameters are not allowed: {extra}')
    if missing:
        fail(f'missing exported parameter states are not allowed: {missing}')
    if out_node is x_node or out_node not in ids or ids[out_node] != len(calls):
        fail(f'output {out_node.name!r} must be the final op value')
    live, stack = set(), [out_node]
    while stack:
        n = stack.pop()
        if n not in live:
            live.add(n)
            stack.extend(n.all_input_nodes)
    dead = [n.name for n in calls if n not in live]
    if dead:
        fail(f'ops not reachable from the output: {dead}')

    blob, layouts = bytearray(), []
    for nid, in_dim, out_dim, w, b in pending:
        wt = w.t().contiguous().reshape(-1).tolist()
        off = len(blob) // 4
        blob += struct.pack(f'<{len(wt)}f', *wt)
        blob += struct.pack(f'<{out_dim}f', *b.reshape(-1).tolist())
        layouts.append({'nodeID': nid, 'inDim': in_dim, 'outDim': out_dim,
                        'weightOffset': off, 'weightCount': len(wt),
                        'biasOffset': off + len(wt)})
    graph = {'format': FORMAT, 'inputDim': input_dim, 'nodes': dag,
             'outputValue': len(calls)}
    return graph, bytes(blob), layouts, aten_ops
