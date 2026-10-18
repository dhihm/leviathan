//! CPU reference interpreter: evaluates an IR graph node by node on float32 tensors.
//!
//! It defines the meaning of every operator and is the oracle for the optimized lowerings; it is
//! written for clarity and checked against PyTorch, not for speed (the Vulkan path is the fast one).

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::ir::{Arg, Graph, Node};
use crate::pack::Packed;
use crate::tensor::{IntTensor, Tensor};

#[derive(Debug, Clone)]
pub enum V {
    T(Tensor),
    I(IntTensor),
    Int(i64),
    Float(f64),
    List(Vec<Rc<V>>),
    None,
}

fn tensor(v: &V) -> Result<&Tensor, String> {
    match v {
        V::T(t) => Ok(t),
        other => Err(format!("expected a float tensor, got {}", kind(other))),
    }
}

fn kind(v: &V) -> &'static str {
    match v {
        V::T(_) => "tensor",
        V::I(_) => "int tensor",
        V::Int(_) => "int",
        V::Float(_) => "float",
        V::List(_) => "list",
        V::None => "none",
    }
}

/// Operators the interpreter implements.
pub const SUPPORTED_OPS: &[&str] = &[
    "aten::sym_size.int",
    "aten::embedding.default",
    "aten::linear.default",
    "aten::squeeze.dim",
    "aten::split_with_sizes.default",
    "aten::chunk.default",
    "aten::cat.default",
    "aten::reshape.default",
    "aten::view.default",
    "aten::repeat_interleave.self_int",
    "aten::sigmoid.default",
    "aten::exp.default",
    "aten::neg.default",
    "aten::softplus.default",
    "aten::mul.Tensor",
    "aten::add.Tensor",
    "aten::to.dtype",
    "aten::_assert_tensor_metadata.default",
    "getitem",
    "lev::rmsnorm.default",
    "lev::rmsnorm_gated.default",
    "lev::swiglu.default",
    "lev::rope.default",
    "lev::causal_conv1d_state.default",
    "lev::gated_delta_rule.default",
    "lev::causal_attention.default",
];

pub fn is_supported(op: &str) -> bool {
    SUPPORTED_OPS.contains(&op) || op.ends_with("getitem")
}

/// Runs the graph. `params` binds parameter names, `inputs` the user inputs by name.
pub fn run(g: &Graph, params: &BTreeMap<String, Packed>, inputs: &BTreeMap<String, Packed>) -> Result<Vec<Rc<V>>, String> {
    let wrap = |p: &Packed| -> Rc<V> {
        Rc::new(match p {
            Packed::F32(t) => V::T(t.clone()),
            Packed::I64(t) => V::I(t.clone()),
        })
    };
    let mut vals: Vec<Rc<V>> = Vec::with_capacity(g.nodes.len());
    let resolve = |a: &Arg, vals: &Vec<Rc<V>>| -> Result<Rc<V>, String> { resolve_arg(a, vals, params, inputs, &wrap) };
    for (i, n) in g.nodes.iter().enumerate() {
        let args: Vec<Rc<V>> = n.args.iter().map(|a| resolve(a, &vals)).collect::<Result<_, _>>()?;
        let v = exec(n, &args, &|a: &Arg| resolve(a, &vals)).map_err(|e| format!("node {i} ({}): {e}", n.op))?;
        vals.push(Rc::new(v));
    }
    g.outputs.iter().map(|a| resolve(a, &vals)).collect()
}

fn resolve_arg(a: &Arg, vals: &[Rc<V>], params: &BTreeMap<String, Packed>, inputs: &BTreeMap<String, Packed>, wrap: &dyn Fn(&Packed) -> Rc<V>) -> Result<Rc<V>, String> {
    Ok(match a {
        Arg::Node(n) => vals.get(*n).ok_or("forward reference")?.clone(),
        Arg::Param(p) => wrap(params.get(p).ok_or(format!("missing parameter {p}"))?),
        Arg::Input(p) => wrap(inputs.get(p).ok_or(format!("missing input {p}"))?),
        Arg::Int(i) => Rc::new(V::Int(*i)),
        Arg::Float(f) => Rc::new(V::Float(*f)),
        Arg::Bool(b) => Rc::new(V::Int(*b as i64)),
        Arg::None => Rc::new(V::None),
        Arg::Str(_) | Arg::Dtype(_) | Arg::Sym(_) => Rc::new(V::None),
        Arg::List(l) => Rc::new(V::List(l.iter().map(|x| resolve_arg(x, vals, params, inputs, wrap)).collect::<Result<_, _>>()?)),
    })
}

fn int(v: &V) -> Result<i64, String> {
    match v {
        V::Int(i) => Ok(*i),
        other => Err(format!("expected an int, got {}", kind(other))),
    }
}

fn float(v: &V) -> Result<f64, String> {
    match v {
        V::Float(f) => Ok(*f),
        V::Int(i) => Ok(*i as f64),
        other => Err(format!("expected a float, got {}", kind(other))),
    }
}

fn axis(dim: i64, rank: usize) -> Result<usize, String> {
    let d = if dim < 0 { dim + rank as i64 } else { dim };
    if d < 0 || d as usize >= rank {
        return Err(format!("dimension {dim} out of range for rank {rank}"));
    }
    Ok(d as usize)
}

fn exec(n: &Node, a: &[Rc<V>], _resolve: &dyn Fn(&Arg) -> Result<Rc<V>, String>) -> Result<V, String> {
    let op = n.op.as_str();
    if op.ends_with("getitem") {
        let V::List(l) = &*a[0] else { return Err("getitem of a non-list".into()) };
        let i = int(&a[1])? as usize;
        return Ok((**l.get(i).ok_or("getitem index out of range")?).clone());
    }
    Ok(match op {
        "aten::sym_size.int" => {
            let d = int(&a[1])?;
            let shape = match &*a[0] {
                V::T(t) => &t.shape,
                V::I(t) => &t.shape,
                other => return Err(format!("sym_size of {}", kind(other))),
            };
            V::Int(shape[axis(d, shape.len())?] as i64)
        }
        "aten::_assert_tensor_metadata.default" => V::None,
        "aten::to.dtype" => (*a[0]).clone(), // float32 only: the dtype argument is informational
        "aten::embedding.default" => {
            let (w, idx) = (tensor(&a[0])?, match &*a[1] {
                V::I(t) => t,
                other => return Err(format!("embedding indices are a {}", kind(other))),
            });
            let d = w.last();
            let mut out = Vec::with_capacity(idx.data.len() * d);
            for &i in &idx.data {
                let i = i as usize;
                if i >= w.shape[0] {
                    return Err(format!("token id {i} out of range"));
                }
                out.extend_from_slice(&w.data[i * d..(i + 1) * d]);
            }
            let mut shape = idx.shape.clone();
            shape.push(d);
            V::T(Tensor::new(shape, out))
        }
        "aten::linear.default" => {
            if !matches!(a.get(2).map(|v| &**v), None | Some(V::None)) {
                return Err("linear with a bias is not supported".into());
            }
            V::T(linear(tensor(&a[0])?, tensor(&a[1])?))
        }
        "aten::squeeze.dim" => {
            let x = tensor(&a[0])?;
            let d = axis(int(&a[1])?, x.shape.len())?;
            let mut shape = x.shape.clone();
            if shape[d] == 1 {
                shape.remove(d);
            }
            V::T(Tensor::new(shape, x.data.clone()))
        }
        "aten::reshape.default" | "aten::view.default" => {
            let x = tensor(&a[0])?;
            let V::List(spec) = &*a[1] else { return Err("reshape needs a size list".into()) };
            let mut shape: Vec<i64> = spec.iter().map(|v| int(v)).collect::<Result<_, _>>()?;
            let known: i64 = shape.iter().filter(|&&d| d >= 0).product();
            if let Some(pos) = shape.iter().position(|&d| d == -1) {
                if known == 0 || x.numel() as i64 % known != 0 {
                    return Err(format!("cannot reshape {:?} to {shape:?}", x.shape));
                }
                shape[pos] = x.numel() as i64 / known;
            }
            V::T(Tensor::new(shape.iter().map(|&d| d as usize).collect(), x.data.clone()))
        }
        "aten::split_with_sizes.default" => {
            let x = tensor(&a[0])?;
            let V::List(sizes) = &*a[1] else { return Err("split sizes".into()) };
            let d = axis(int(&a[2])?, x.shape.len())?;
            let sizes: Vec<usize> = sizes.iter().map(|v| int(v).map(|i| i as usize)).collect::<Result<_, _>>()?;
            V::List(split(x, &sizes, d)?.into_iter().map(|t| Rc::new(V::T(t))).collect())
        }
        "aten::chunk.default" => {
            let x = tensor(&a[0])?;
            let chunks = int(&a[1])? as usize;
            let d = axis(int(&a[2])?, x.shape.len())?;
            let size = x.shape[d].div_ceil(chunks);
            let mut sizes = Vec::new();
            let mut left = x.shape[d];
            while left > 0 {
                let s = size.min(left);
                sizes.push(s);
                left -= s;
            }
            V::List(split(x, &sizes, d)?.into_iter().map(|t| Rc::new(V::T(t))).collect())
        }
        "aten::cat.default" => {
            let V::List(items) = &*a[0] else { return Err("cat needs a list".into()) };
            let ts: Vec<&Tensor> = items.iter().map(|v| tensor(v)).collect::<Result<_, _>>()?;
            let dim = match a.get(1) {
                Some(v) => int(v)?,
                None => 0,
            };
            V::T(cat(&ts, axis(dim, ts[0].shape.len())?)?)
        }
        "aten::repeat_interleave.self_int" => {
            let x = tensor(&a[0])?;
            let rep = int(&a[1])? as usize;
            let d = axis(int(&a[2])?, x.shape.len())?;
            let (outer, len, inner) = x.split_dims(d);
            let mut out = Vec::with_capacity(x.numel() * rep);
            for o in 0..outer {
                for i in 0..len {
                    let row = &x.data[(o * len + i) * inner..(o * len + i + 1) * inner];
                    for _ in 0..rep {
                        out.extend_from_slice(row);
                    }
                }
            }
            let mut shape = x.shape.clone();
            shape[d] *= rep;
            V::T(Tensor::new(shape, out))
        }
        "aten::sigmoid.default" => V::T(map(tensor(&a[0])?, |x| 1.0 / (1.0 + (-x).exp()))),
        "aten::exp.default" => V::T(map(tensor(&a[0])?, f32::exp)),
        "aten::neg.default" => V::T(map(tensor(&a[0])?, |x| -x)),
        "aten::softplus.default" => V::T(map(tensor(&a[0])?, |x| if x > 20.0 { x } else { x.exp().ln_1p() })),
        "aten::mul.Tensor" => V::T(binary(&a[0], &a[1], |x, y| x * y)?),
        "aten::add.Tensor" => {
            if let Some(alpha) = n.kwarg("alpha") {
                if !matches!(alpha, Arg::Float(f) if *f == 1.0) && !matches!(alpha, Arg::Int(1)) {
                    return Err("add with alpha != 1 is not supported".into());
                }
            }
            V::T(binary(&a[0], &a[1], |x, y| x + y)?)
        }
        "lev::rmsnorm.default" => {
            let zero_centered = int(&a[3])? != 0;
            V::T(rmsnorm(tensor(&a[0])?, tensor(&a[1])?, float(&a[2])? as f32, zero_centered))
        }
        "lev::rmsnorm_gated.default" => V::T(rmsnorm_gated(tensor(&a[0])?, tensor(&a[1])?, tensor(&a[2])?, float(&a[3])? as f32)),
        "lev::swiglu.default" => {
            let (g, u) = (tensor(&a[0])?, tensor(&a[1])?);
            if g.shape != u.shape {
                return Err("swiglu shapes differ".into());
            }
            V::T(Tensor::new(g.shape.clone(), g.data.iter().zip(&u.data).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect()))
        }
        "lev::rope.default" => V::T(rope(tensor(&a[0])?, tensor(&a[1])?, tensor(&a[2])?)),
        "lev::causal_conv1d_state.default" => {
            let (y, s) = conv_state(tensor(&a[0])?, tensor(&a[1])?, tensor(&a[2])?);
            V::List(vec![Rc::new(V::T(y)), Rc::new(V::T(s))])
        }
        "lev::gated_delta_rule.default" => {
            let t: Vec<&Tensor> = a[..6].iter().map(|v| tensor(v)).collect::<Result<_, _>>()?;
            let (o, s) = gated_delta_rule(t[0], t[1], t[2], t[3], t[4], t[5]);
            V::List(vec![Rc::new(V::T(o)), Rc::new(V::T(s))])
        }
        "lev::causal_attention.default" => {
            V::T(causal_attention(tensor(&a[0])?, tensor(&a[1])?, tensor(&a[2])?, int(&a[3])? as usize, float(&a[4])? as f32))
        }
        other => return Err(format!("unsupported operator {other}")),
    })
}

fn map(x: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    Tensor { shape: x.shape.clone(), data: x.data.iter().map(|&v| f(v)).collect() }
}

/// Elementwise binary operation with PyTorch broadcasting; either side may be a scalar.
fn binary(a: &V, b: &V, f: impl Fn(f32, f32) -> f32) -> Result<Tensor, String> {
    let scalar = |v: &V| -> Option<f32> {
        match v {
            V::Float(x) => Some(*x as f32),
            V::Int(x) => Some(*x as f32),
            _ => None,
        }
    };
    match (a, b) {
        (V::T(x), s) if scalar(s).is_some() => {
            let s = scalar(s).unwrap();
            Ok(Tensor { shape: x.shape.clone(), data: x.data.iter().map(|&v| f(v, s)).collect() })
        }
        (s, V::T(y)) if scalar(s).is_some() => {
            let s = scalar(s).unwrap();
            Ok(Tensor { shape: y.shape.clone(), data: y.data.iter().map(|&v| f(s, v)).collect() })
        }
        (V::T(x), V::T(y)) => {
            if x.shape == y.shape {
                return Ok(Tensor { shape: x.shape.clone(), data: x.data.iter().zip(&y.data).map(|(&p, &q)| f(p, q)).collect() });
            }
            let rank = x.shape.len().max(y.shape.len());
            let dim = |t: &Tensor, i: usize| -> usize {
                let off = rank - t.shape.len();
                if i < off {
                    1
                } else {
                    t.shape[i - off]
                }
            };
            let out_shape: Vec<usize> = (0..rank)
                .map(|i| {
                    let (p, q) = (dim(x, i), dim(y, i));
                    if p == q || q == 1 {
                        Ok(p)
                    } else if p == 1 {
                        Ok(q)
                    } else {
                        Err(format!("cannot broadcast {:?} with {:?}", x.shape, y.shape))
                    }
                })
                .collect::<Result<_, _>>()?;
            let strides = |t: &Tensor| -> Vec<usize> {
                let mut s = vec![0; rank];
                let mut acc = 1;
                for i in (0..rank).rev() {
                    let d = dim(t, i);
                    s[i] = if d == 1 { 0 } else { acc };
                    acc *= d;
                }
                s
            };
            let (sx, sy) = (strides(x), strides(y));
            let total: usize = out_shape.iter().product();
            let mut out = Vec::with_capacity(total);
            let mut idx = vec![0usize; rank];
            for _ in 0..total {
                let (mut ox, mut oy) = (0, 0);
                for i in 0..rank {
                    ox += idx[i] * sx[i];
                    oy += idx[i] * sy[i];
                }
                out.push(f(x.data[ox], y.data[oy]));
                for i in (0..rank).rev() {
                    idx[i] += 1;
                    if idx[i] < out_shape[i] {
                        break;
                    }
                    idx[i] = 0;
                }
            }
            Ok(Tensor { shape: out_shape, data: out })
        }
        _ => Err("binary operation on unsupported operands".into()),
    }
}

fn linear(x: &Tensor, w: &Tensor) -> Tensor {
    let (n, k) = (w.shape[0], w.shape[1]);
    assert_eq!(x.last(), k, "linear: input width {} != weight width {k}", x.last());
    let rows = x.numel() / k;
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
        let xr = &x.data[r * k..(r + 1) * k];
        for (j, o) in out[r * n..(r + 1) * n].iter_mut().enumerate() {
            let wr = &w.data[j * k..(j + 1) * k];
            *o = xr.iter().zip(wr).map(|(&p, &q)| p * q).sum();
        }
    }
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = n;
    Tensor::new(shape, out)
}

fn split(x: &Tensor, sizes: &[usize], d: usize) -> Result<Vec<Tensor>, String> {
    let (outer, len, inner) = x.split_dims(d);
    if sizes.iter().sum::<usize>() != len {
        return Err(format!("split sizes {sizes:?} do not add up to {len}"));
    }
    let mut out = Vec::new();
    let mut start = 0;
    for &s in sizes {
        let mut data = Vec::with_capacity(outer * s * inner);
        for o in 0..outer {
            data.extend_from_slice(&x.data[(o * len + start) * inner..(o * len + start + s) * inner]);
        }
        let mut shape = x.shape.clone();
        shape[d] = s;
        out.push(Tensor::new(shape, data));
        start += s;
    }
    Ok(out)
}

fn cat(ts: &[&Tensor], d: usize) -> Result<Tensor, String> {
    let first = ts[0];
    let (outer, _, inner) = first.split_dims(d);
    let total: usize = ts.iter().map(|t| t.shape[d]).sum();
    for t in ts {
        let same = t.shape.len() == first.shape.len() && t.shape.iter().enumerate().all(|(i, &s)| i == d || s == first.shape[i]);
        if !same {
            return Err(format!("cat of shapes {:?} and {:?} along {d}", first.shape, t.shape));
        }
    }
    let mut data = Vec::with_capacity(outer * total * inner);
    for o in 0..outer {
        for t in ts {
            let len = t.shape[d];
            data.extend_from_slice(&t.data[o * len * inner..(o + 1) * len * inner]);
        }
    }
    let mut shape = first.shape.clone();
    shape[d] = total;
    Ok(Tensor::new(shape, data))
}

fn rmsnorm(x: &Tensor, w: &Tensor, eps: f32, zero_centered: bool) -> Tensor {
    let d = x.last();
    let mut out = Vec::with_capacity(x.numel());
    let add = if zero_centered { 1.0 } else { 0.0 };
    for row in x.data.chunks_exact(d) {
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        out.extend(row.iter().zip(&w.data).map(|(&v, &g)| v * inv * (g + add)));
    }
    Tensor::new(x.shape.clone(), out)
}

fn rmsnorm_gated(x: &Tensor, z: &Tensor, w: &Tensor, eps: f32) -> Tensor {
    let d = x.last();
    let mut out = Vec::with_capacity(x.numel());
    for (row, zrow) in x.data.chunks_exact(d).zip(z.data.chunks_exact(d)) {
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        out.extend(row.iter().zip(zrow).zip(&w.data).map(|((&v, &g), &wt)| wt * (v * inv) * (g / (1.0 + (-g).exp()))));
    }
    Tensor::new(x.shape.clone(), out)
}

fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let (t, h, d) = (x.shape[0], x.shape[1], x.shape[2]);
    let rd = cos.shape[1];
    let half = rd / 2;
    let mut out = x.data.clone();
    for ti in 0..t {
        let (c, s) = (&cos.data[ti * rd..(ti + 1) * rd], &sin.data[ti * rd..(ti + 1) * rd]);
        for hi in 0..h {
            let base = (ti * h + hi) * d;
            for i in 0..half {
                let (x1, x2) = (x.data[base + i], x.data[base + i + half]);
                out[base + i] = x1 * c[i] - x2 * s[i];
                out[base + i + half] = x2 * c[i + half] + x1 * s[i + half];
            }
        }
    }
    Tensor::new(x.shape.clone(), out)
}

/// y = silu(causal depthwise conv over cat(state, x)), new state = last K-1 rows.
fn conv_state(x: &Tensor, state: &Tensor, w: &Tensor) -> (Tensor, Tensor) {
    let (t, c) = (x.shape[0], x.shape[1]);
    let k = w.shape[1];
    let km1 = k - 1;
    let row = |r: usize| -> &[f32] {
        if r < km1 {
            &state.data[r * c..(r + 1) * c]
        } else {
            &x.data[(r - km1) * c..(r - km1 + 1) * c]
        }
    };
    let mut y = vec![0f32; t * c];
    for ti in 0..t {
        for ci in 0..c {
            let mut acc = 0f32;
            for i in 0..k {
                acc += row(ti + i)[ci] * w.data[ci * k + i];
            }
            y[ti * c + ci] = acc / (1.0 + (-acc).exp());
        }
    }
    let mut ns = Vec::with_capacity(km1 * c);
    for r in t..t + km1 {
        ns.extend_from_slice(row(r));
    }
    (Tensor::new(vec![t, c], y), Tensor::new(vec![km1, c], ns))
}

fn l2norm(v: &[f32]) -> Vec<f32> {
    let inv = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
    v.iter().map(|x| x * inv).collect()
}

/// Recurrent gated delta rule, one token at a time (see tools/model_ir/lev_ops.py).
fn gated_delta_rule(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &Tensor) -> (Tensor, Tensor) {
    let (t, h, dk) = (q.shape[0], q.shape[1], q.shape[2]);
    let dv = v.shape[2];
    let scale = (dk as f32).powf(-0.5);
    let mut s = state.data.clone(); // [H, dk, dv]
    let mut out = vec![0f32; t * h * dv];
    for ti in 0..t {
        for hi in 0..h {
            let qh: Vec<f32> = l2norm(&q.data[(ti * h + hi) * dk..(ti * h + hi + 1) * dk]).iter().map(|x| x * scale).collect();
            let kh = l2norm(&k.data[(ti * h + hi) * dk..(ti * h + hi + 1) * dk]);
            let vh = &v.data[(ti * h + hi) * dv..(ti * h + hi + 1) * dv];
            let decay = g.data[ti * h + hi].exp();
            let b = beta.data[ti * h + hi];
            let sh = &mut s[hi * dk * dv..(hi + 1) * dk * dv];
            for x in sh.iter_mut() {
                *x *= decay;
            }
            let mut kv = vec![0f32; dv];
            for i in 0..dk {
                for j in 0..dv {
                    kv[j] += sh[i * dv + j] * kh[i];
                }
            }
            let delta: Vec<f32> = (0..dv).map(|j| (vh[j] - kv[j]) * b).collect();
            let o = &mut out[(ti * h + hi) * dv..(ti * h + hi + 1) * dv];
            for i in 0..dk {
                for j in 0..dv {
                    sh[i * dv + j] += kh[i] * delta[j];
                    o[j] += sh[i * dv + j] * qh[i];
                }
            }
        }
    }
    (Tensor::new(vec![t, h, dv], out), Tensor::new(state.shape.clone(), s))
}

fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, past: usize, scale: f32) -> Tensor {
    let (t, hq, d) = (q.shape[0], q.shape[1], q.shape[2]);
    let (s, hkv) = (k.shape[0], k.shape[1]);
    let rep = hq / hkv;
    let mut out = vec![0f32; t * hq * d];
    let mut scores = vec![0f32; s];
    for ti in 0..t {
        let limit = (past + ti + 1).min(s);
        for h in 0..hq {
            let kvh = h / rep;
            let qv = &q.data[(ti * hq + h) * d..(ti * hq + h + 1) * d];
            let mut max = f32::NEG_INFINITY;
            for p in 0..limit {
                let kv = &k.data[(p * hkv + kvh) * d..(p * hkv + kvh + 1) * d];
                scores[p] = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
                max = max.max(scores[p]);
            }
            let mut sum = 0f32;
            for sc in &mut scores[..limit] {
                *sc = (*sc - max).exp();
                sum += *sc;
            }
            let o = &mut out[(ti * hq + h) * d..(ti * hq + h + 1) * d];
            for p in 0..limit {
                let w = scores[p] / sum;
                let vv = &v.data[(p * hkv + kvh) * d..(p * hkv + kvh + 1) * d];
                for j in 0..d {
                    o[j] += w * vv[j];
                }
            }
        }
    }
    Tensor::new(vec![t, hq, d], out)
}
