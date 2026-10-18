//! Fusion by pattern rewriting: turns the (retargeted) model IR into a list of kernel operations.
//!
//! Each rule starts at a head node, checks the shape of the subgraph around it (operators, use
//! counts, which parameter feeds which input) and *claims* those nodes. The result is complete
//! only if every node of the graph was claimed, so an operator that no rule covers is an error
//! instead of a silently dropped computation. Buffer slots are decided from the consumer that
//! reads a value (a convolution reads the QKV projection, the delta kernel reads alpha, beta and
//! z), never from parameter names.
//!
//! The kernel operations map one to one onto the emitters of `gpu.rs`; `gpu.rs` checks that the
//! lowered command stream equals the hand-written one.

use lev_ir::bind;
use lev_ir::ir::{Arg, Graph};

/// Scratch buffer a kernel operation reads or writes (fields of the GPU resources).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Mixed,
    Down,
    Gate,
    Up,
    Qkv,
    Z,
    Beta,
    Alpha,
    Gated,
    Qfull,
    Kin,
    Vin,
    AttnOut,
    Logits,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KOp {
    /// Token embedding into the residual stream.
    Embed,
    /// `x += add` (when given), RMS norm with `weight`, int8 quantization of the result for the
    /// matvecs that follow. `keep_hidden` also keeps the f32 normalized values (the MTP head reads
    /// the final hidden state).
    NormQuant { weight: String, add: Option<Slot>, keep_hidden: bool },
    /// Int8 quantization of a computed activation.
    Quant { src: Slot },
    /// `out = W * quantized activation` for the GGUF tensor `weight`.
    Matvec { weight: String, out: Slot },
    SsmConv { layer: usize, weight: String },
    /// L2 normalization (and scaling) of the q and k heads of the convolution output.
    SsmQkNorm,
    /// Gated delta rule with the decay and beta gates computed inside, then the gated RMS norm.
    SsmDelta { layer: usize, dt_bias: String, a: String, norm: String },
    /// q/k norms, rope and the KV cache write.
    AttnPrep { layer: usize, q_norm: String, k_norm: String },
    /// Attention over the cache followed by the output gate.
    AttnDecode { layer: usize },
    SwigluQuant,
    /// Ragged batch: causal convolution on the state-pool slot of each column (one column per slot).
    SsmConvBatch { layer: usize, weight: String },
    /// Ragged batch: gated delta rule on the state-pool slot of each column, gates and gated norm inside.
    SsmDeltaBatch { layer: usize, dt_bias: String, a: String, norm: String },
    /// Ragged batch: q/k norms, rope and the write of K and V into each column's slab.
    AttnPrepBatch { layer: usize, q_norm: String, k_norm: String },
    /// Ragged batch: attention of each column over its own slab, then the output gate.
    AttnDecodeBatch { layer: usize },
    /// Paged KV: q/k norms, rope and the write of K and V into the block pool.
    AttnPrepPaged { layer: usize, q_norm: String, k_norm: String },
    /// Paged KV: attention over the blocks of the column's sequence, then the output gate.
    AttnDecodePaged { layer: usize },
    /// Greedy token and margin of the logits (runtime epilogue, not part of the IR).
    Argmax,
}

struct Lower<'g> {
    g: &'g Graph,
    users: Vec<Vec<usize>>,
    claimed: Vec<bool>,
    ops: Vec<(usize, usize, KOp)>,
}

fn err<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(msg.into())
}

impl<'g> Lower<'g> {
    fn op(&self, i: usize) -> &str {
        &self.g.nodes[i].op
    }

    fn arg(&self, i: usize, k: usize) -> Result<&Arg, String> {
        self.g.nodes[i].args.get(k).ok_or(format!("node {i} ({}) has no argument {k}", self.op(i)))
    }

    fn node_arg(&self, i: usize, k: usize) -> Result<usize, String> {
        match self.arg(i, k)? {
            Arg::Node(n) => Ok(*n),
            other => err(format!("node {i} ({}): argument {k} is {other:?}, expected a node", self.op(i))),
        }
    }

    fn param_arg(&self, i: usize, k: usize) -> Result<&str, String> {
        match self.arg(i, k)? {
            Arg::Param(p) => Ok(p),
            other => err(format!("node {i} ({}): argument {k} is {other:?}, expected a parameter", self.op(i))),
        }
    }

    fn gguf(&self, param: &str) -> Result<String, String> {
        bind::to_gguf(param).map(|(n, _)| n).ok_or(format!("no GGUF binding for {param}"))
    }

    fn expect(&self, i: usize, op: &str) -> Result<usize, String> {
        if self.op(i) == op {
            Ok(i)
        } else {
            err(format!("node {i} is {}, expected {op}", self.op(i)))
        }
    }

    /// The node feeding argument `k` of node `i`, required to be `op`.
    fn feeder(&self, i: usize, k: usize, op: &str) -> Result<usize, String> {
        let n = self.node_arg(i, k)?;
        self.expect(n, op)
    }

    fn getitem(&self, i: usize, index: i64) -> Result<usize, String> {
        let src = self.expect(i, "getitem")?;
        match self.arg(src, 1)? {
            Arg::Int(x) if *x == index => self.node_arg(src, 0),
            other => err(format!("node {i}: getitem {other:?}, expected index {index}")),
        }
    }

    fn sole_user(&self, i: usize) -> Result<usize, String> {
        match self.users[i].as_slice() {
            [u] => Ok(*u),
            other => err(format!("node {i} ({}) has {} users, expected one", self.op(i), other.len())),
        }
    }

    fn claim(&mut self, nodes: &[usize]) -> Result<(), String> {
        for &n in nodes {
            if self.claimed[n] {
                return err(format!("node {n} ({}) is claimed twice", self.op(n)));
            }
            self.claimed[n] = true;
        }
        Ok(())
    }

    fn emit(&mut self, anchor: usize, seq: usize, op: KOp) {
        self.ops.push((anchor, seq, op));
    }

    /// Layer index of a state input (`states_N`: two states per layer).
    fn state_layer(&self, a: &Arg) -> Result<usize, String> {
        match a {
            Arg::Input(name) => name.strip_prefix("states_").or_else(|| name.strip_prefix("pools_")).and_then(|n| n.parse::<usize>().ok()).map(|n| n / 2).ok_or(format!("{name} is not a state input")),
            other => err(format!("expected a state input, got {other:?}")),
        }
    }

    /// Where the output of linear node `i` lives, decided by the operation that reads it.
    fn linear_slot(&self, i: usize) -> Result<Slot, String> {
        let users = &self.users[i];
        if users.is_empty() {
            return if self.g.output_nodes().contains(&i) { Ok(Slot::Logits) } else { err(format!("linear {i} is unused")) };
        }
        let u = users[0];
        Ok(match self.op(u) {
            "lev::causal_conv1d_state.default" | "lev::ssm_conv_ragged.default" => Slot::Qkv,
            "aten::sigmoid.default" => Slot::Beta,
            "aten::to.dtype" => Slot::Alpha,
            "aten::reshape.default" => {
                // z: reshape feeding the gate input of the gated norm
                let r = self.sole_user(u)?;
                self.expect(r, "lev::rmsnorm_gated.default")?;
                Slot::Z
            }
            "aten::view.default" => {
                let v = self.sole_user(u)?;
                match self.op(v) {
                    "aten::chunk.default" => Slot::Qfull,
                    "lev::rmsnorm.default" => Slot::Kin,
                    "aten::cat.default" | "lev::slab_kv_write.default" | "lev::paged_kv_write.default" => Slot::Vin,
                    other => return err(format!("linear {i}: its view feeds {other}")),
                }
            }
            "lev::swiglu.default" => {
                if self.node_arg(u, 0)? == i {
                    Slot::Gate
                } else {
                    Slot::Up
                }
            }
            "aten::add.Tensor" => {
                // The residual branch: the FFN output (input from swiglu) or the mixer output.
                let input = self.node_arg(i, 0)?;
                if self.op(input) == "lev::swiglu.default" {
                    Slot::Down
                } else {
                    Slot::Mixed
                }
            }
            other => return err(format!("linear {i}: its output feeds {other}, which no kernel reads")),
        })
    }

    fn slot_of_branch(&self, add: usize) -> Result<Slot, String> {
        let b = self.node_arg(add, 1)?;
        self.expect(b, "aten::linear.default")?;
        self.linear_slot(b)
    }

    // ---- rules ----

    /// rmsnorm whose users are all linear layers: norm + quantize, then one matvec per user.
    /// An `add` feeding the norm (the residual) is folded into the norm kernel.
    fn rule_norm_linears(&mut self, i: usize) -> Result<bool, String> {
        let users = self.users[i].clone();
        if users.is_empty() || users.iter().any(|&u| self.op(u) != "aten::linear.default") {
            return Ok(false);
        }
        let x = self.node_arg(i, 0)?;
        let mut add = None;
        let mut claim = vec![i];
        if self.op(x) == "aten::add.Tensor" {
            // x = add(residual, branch): the residual stream lives in place, the branch in a slot.
            let residual = self.node_arg(x, 0)?;
            if !matches!(self.op(residual), "aten::embedding.default" | "aten::add.Tensor") {
                return err(format!("add {x}: the residual comes from {}", self.op(residual)));
            }
            add = Some(self.slot_of_branch(x)?);
            claim.push(x);
        } else if self.op(x) != "aten::embedding.default" {
            return err(format!("norm {i}: its input is {}", self.op(x)));
        }
        let weight = self.gguf(self.param_arg(i, 1)?)?;
        let keep_hidden = users.iter().any(|&u| self.g.output_nodes().contains(&u));
        self.emit(i, 0, KOp::NormQuant { weight, add, keep_hidden });
        for (k, &u) in users.iter().enumerate() {
            let w = self.gguf(self.param_arg(u, 1)?)?;
            let out = self.linear_slot(u)?;
            self.emit(u, k, KOp::Matvec { weight: w, out });
            claim.push(u);
        }
        self.claim(&claim)?;
        Ok(true)
    }

    fn rule_conv(&mut self, i: usize) -> Result<(), String> {
        let ragged = self.op(i) == "lev::ssm_conv_ragged.default";
        let layer = self.state_layer(self.arg(i, 1)?)?;
        let squeeze = self.feeder(i, if ragged { 3 } else { 2 }, "aten::squeeze.dim")?;
        let weight = self.gguf(self.param_arg(squeeze, 0)?)?;
        let out = self.users[i].iter().copied().find(|&u| self.getitem(u, 0).is_ok()).ok_or("conv: no output getitem")?;
        let state = self.users[i].iter().copied().find(|&u| self.getitem(u, 1).is_ok()).ok_or("conv: no state getitem")?;
        self.emit(i, 0, if ragged { KOp::SsmConvBatch { layer, weight } } else { KOp::SsmConv { layer, weight } });
        self.claim(&[i, squeeze, out, state])
    }

    fn rule_delta(&mut self, d: usize) -> Result<(), String> {
        let ragged = self.op(d) == "lev::gated_delta_ragged.default";
        let conv_op = if ragged { "lev::ssm_conv_ragged.default" } else { "lev::causal_conv1d_state.default" };
        let state_layer = self.state_layer(self.arg(d, 5)?)?;
        // q and k: tile(reshape(getitem(split(getitem(conv, 0)), 0 or 1)))
        let mut claim = vec![d];
        let mut split = None;
        let mut conv_out = None;
        for (k, which) in [(0usize, 0i64), (1, 1)] {
            let tile = self.feeder(d, k, "lev::tile_heads.default")?;
            let rs = self.feeder(tile, 0, "aten::reshape.default")?;
            let gi = self.node_arg(rs, 0)?;
            let sp = self.getitem(gi, which)?;
            self.expect(sp, "aten::split_with_sizes.default")?;
            let co = self.node_arg(sp, 0)?;
            if *split.get_or_insert(sp) != sp || *conv_out.get_or_insert(co) != co {
                return err("delta: q and k come from different splits");
            }
            claim.extend([tile, rs, gi]);
        }
        let split = split.unwrap();
        let conv_out = conv_out.unwrap();
        let conv = self.getitem(conv_out, 0)?;
        self.expect(conv, conv_op)?;
        if self.state_layer(self.arg(conv, 1)?)? != state_layer {
            return err("delta and conv state belong to different layers");
        }
        // v: reshape(getitem(split, 2))
        let vr = self.feeder(d, 2, "aten::reshape.default")?;
        let vg = self.node_arg(vr, 0)?;
        if self.getitem(vg, 2)? != split {
            return err("delta: v does not come from the split of the convolution output");
        }
        claim.extend([vr, vg, split]);
        // decay gate: mul(ssm_a, softplus(add(to(alpha), dt_bias)))
        let gmul = self.feeder(d, 3, "aten::mul.Tensor")?;
        let a_param = self.gguf(self.param_arg(gmul, 0)?)?;
        let sp = self.feeder(gmul, 1, "aten::softplus.default")?;
        let add = self.feeder(sp, 0, "aten::add.Tensor")?;
        let to = self.feeder(add, 0, "aten::to.dtype")?;
        let dt_bias = self.gguf(self.param_arg(add, 1)?)?;
        let alpha = self.node_arg(to, 0)?;
        self.expect(alpha, "aten::linear.default")?;
        claim.extend([gmul, sp, add, to]);
        // beta: sigmoid(linear)
        let sig = self.feeder(d, 4, "aten::sigmoid.default")?;
        let beta = self.node_arg(sig, 0)?;
        self.expect(beta, "aten::linear.default")?;
        claim.push(sig);
        // outputs: getitem 0 -> reshape -> rmsnorm_gated(.., reshape(z)) -> reshape -> out_proj
        let out_g = self.users[d].iter().copied().find(|&u| self.getitem(u, 0).is_ok()).ok_or("delta: no output getitem")?;
        let st_g = self.users[d].iter().copied().find(|&u| self.getitem(u, 1).is_ok()).ok_or("delta: no state getitem")?;
        let r1 = self.sole_user(out_g)?;
        self.expect(r1, "aten::reshape.default")?;
        let gn = self.sole_user(r1)?;
        self.expect(gn, "lev::rmsnorm_gated.default")?;
        let zr = self.feeder(gn, 1, "aten::reshape.default")?;
        let z = self.node_arg(zr, 0)?;
        self.expect(z, "aten::linear.default")?;
        let norm = self.gguf(self.param_arg(gn, 2)?)?;
        let r3 = self.sole_user(gn)?;
        self.expect(r3, "aten::reshape.default")?;
        claim.extend([out_g, st_g, r1, gn, zr, r3]);
        self.emit(d, 0, KOp::SsmQkNorm);
        self.emit(d, 1, if ragged { KOp::SsmDeltaBatch { layer: state_layer, dt_bias, a: a_param, norm } } else { KOp::SsmDelta { layer: state_layer, dt_bias, a: a_param, norm } });
        self.claim(&claim)
    }

    fn rule_attention(&mut self, a: usize) -> Result<(), String> {
        let (k_all, v_all) = (self.feeder(a, 1, "aten::cat.default")?, self.feeder(a, 2, "aten::cat.default")?);
        let layer = {
            let Arg::List(l) = self.arg(k_all, 0)? else { return err("cat without a list") };
            self.state_layer(&l[0])?
        };
        let list_node = |s: &Self, cat: usize| -> Result<usize, String> {
            let Arg::List(l) = s.arg(cat, 0)? else { return err("cat without a list") };
            match l.get(1) {
                Some(Arg::Node(n)) => Ok(*n),
                other => err(format!("cat {cat}: second operand is {other:?}")),
            }
        };
        let rope_k = list_node(self, k_all)?;
        self.expect(rope_k, "lev::rope.default")?;
        let v_view = list_node(self, v_all)?;
        self.expect(v_view, "aten::view.default")?;
        let k_norm = self.feeder(rope_k, 0, "lev::rmsnorm.default")?;
        let k_view = self.feeder(k_norm, 0, "aten::view.default")?;
        let rope_q = self.feeder(a, 0, "lev::rope.default")?;
        let q_norm = self.feeder(rope_q, 0, "lev::rmsnorm.default")?;
        let q_item = self.node_arg(q_norm, 0)?;
        let chunk = {
            let c = self.getitem(q_item, 0)?;
            self.expect(c, "aten::chunk.default")?
        };
        let q_view = self.feeder(chunk, 0, "aten::view.default")?;
        // output gate: mul(reshape(attention), sigmoid(reshape(getitem(chunk, 1))))
        let out_r = self.sole_user(a)?;
        self.expect(out_r, "aten::reshape.default")?;
        let mul = self.sole_user(out_r)?;
        self.expect(mul, "aten::mul.Tensor")?;
        let sig = self.feeder(mul, 1, "aten::sigmoid.default")?;
        let gate_r = self.feeder(sig, 0, "aten::reshape.default")?;
        let gate_i = self.node_arg(gate_r, 0)?;
        if self.getitem(gate_i, 1)? != chunk {
            return err("attention gate does not come from the q projection");
        }
        let mut items: Vec<usize> = self.users[chunk].clone();
        items.sort_unstable();
        let (qn, kn) = (self.param_arg(q_norm, 1)?.to_string(), self.param_arg(k_norm, 1)?.to_string());
        let (q_norm_w, k_norm_w) = (self.gguf(&qn)?, self.gguf(&kn)?);
        self.emit(a, 0, KOp::AttnPrep { layer, q_norm: q_norm_w, k_norm: k_norm_w });
        self.emit(a, 1, KOp::AttnDecode { layer });
        let mut claim = vec![a, k_all, v_all, rope_k, v_view, k_norm, k_view, rope_q, q_norm, q_view, chunk, out_r, mul, sig, gate_r];
        claim.extend(items);
        self.claim(&claim)
    }

    /// Ragged attention: q/k norm and rope, the write of K and V into the slab, attention over the
    /// slab and the output gate.
    fn rule_slab_attention(&mut self, a: usize) -> Result<(), String> {
        self.rule_attention_over_pool(a, false)
    }

    fn rule_paged_attention(&mut self, a: usize) -> Result<(), String> {
        self.rule_attention_over_pool(a, true)
    }

    fn rule_attention_over_pool(&mut self, a: usize, paged: bool) -> Result<(), String> {
        let (kg, vg) = (self.node_arg(a, 1)?, self.node_arg(a, 2)?);
        let kv_write = self.getitem(kg, 0)?;
        self.expect(kv_write, if paged { "lev::paged_kv_write.default" } else { "lev::slab_kv_write.default" })?;
        if self.getitem(vg, 1)? != kv_write {
            return err("slab attention reads the K and V pools of different writes");
        }
        let layer = self.state_layer(self.arg(kv_write, 2)?)?;
        let rope_k = self.feeder(kv_write, 0, "lev::rope.default")?;
        let v_view = self.feeder(kv_write, 1, "aten::view.default")?;
        let k_norm = self.feeder(rope_k, 0, "lev::rmsnorm.default")?;
        let k_view = self.feeder(k_norm, 0, "aten::view.default")?;
        let rope_q = self.feeder(a, 0, "lev::rope.default")?;
        let q_norm = self.feeder(rope_q, 0, "lev::rmsnorm.default")?;
        let q_item = self.node_arg(q_norm, 0)?;
        let chunk = {
            let c = self.getitem(q_item, 0)?;
            self.expect(c, "aten::chunk.default")?
        };
        let q_view = self.feeder(chunk, 0, "aten::view.default")?;
        let out_r = self.sole_user(a)?;
        self.expect(out_r, "aten::reshape.default")?;
        let mul = self.sole_user(out_r)?;
        self.expect(mul, "aten::mul.Tensor")?;
        let sig = self.feeder(mul, 1, "aten::sigmoid.default")?;
        let gate_r = self.feeder(sig, 0, "aten::reshape.default")?;
        let gate_i = self.node_arg(gate_r, 0)?;
        if self.getitem(gate_i, 1)? != chunk {
            return err("attention gate does not come from the q projection");
        }
        let mut items: Vec<usize> = self.users[chunk].clone();
        items.sort_unstable();
        let (qn, kn) = (self.param_arg(q_norm, 1)?.to_string(), self.param_arg(k_norm, 1)?.to_string());
        let (q_norm_w, k_norm_w) = (self.gguf(&qn)?, self.gguf(&kn)?);
        if paged {
            self.emit(a, 0, KOp::AttnPrepPaged { layer, q_norm: q_norm_w, k_norm: k_norm_w });
            self.emit(a, 1, KOp::AttnDecodePaged { layer });
        } else {
            self.emit(a, 0, KOp::AttnPrepBatch { layer, q_norm: q_norm_w, k_norm: k_norm_w });
            self.emit(a, 1, KOp::AttnDecodeBatch { layer });
        }
        let mut claim = vec![a, kg, vg, kv_write, rope_k, v_view, k_norm, k_view, rope_q, q_norm, q_view, chunk, out_r, mul, sig, gate_r];
        claim.extend(items);
        self.claim(&claim)
    }

    fn rule_swiglu(&mut self, s: usize) -> Result<(), String> {
        let down = self.sole_user(s)?;
        self.expect(down, "aten::linear.default")?;
        let w = self.gguf(self.param_arg(down, 1)?)?;
        let out = self.linear_slot(down)?;
        self.emit(s, 0, KOp::SwigluQuant);
        self.emit(down, 0, KOp::Matvec { weight: w, out });
        self.claim(&[s, down])
    }

    /// A linear whose input was computed (not produced by a norm or swiglu kernel) needs its
    /// activation quantized first.
    fn rule_linear(&mut self, l: usize) -> Result<(), String> {
        let src = self.node_arg(l, 0)?;
        let slot = match self.op(src) {
            "aten::reshape.default" => Slot::Gated,
            "aten::mul.Tensor" => Slot::AttnOut,
            other => return err(format!("linear {l}: input comes from {other}")),
        };
        let w = self.gguf(self.param_arg(l, 1)?)?;
        let out = self.linear_slot(l)?;
        self.emit(l, 0, KOp::Quant { src: slot });
        self.emit(l, 1, KOp::Matvec { weight: w, out });
        self.claim(&[l])
    }
}

/// Lowers a graph that went through `pass::retarget_gguf`.
pub fn lower(g: &Graph) -> Result<Vec<KOp>, String> {
    let mut l = Lower { g, users: g.users(), claimed: vec![false; g.nodes.len()], ops: Vec::new() };
    // The scan claims whole subgraphs; the order of the rules inside one pass does not matter
    // because every rule starts at a distinct kind of head node.
    for i in 0..g.nodes.len() {
        match g.nodes[i].op.as_str() {
            "aten::sym_size.int" => l.claim(&[i])?,
            "aten::embedding.default" => {
                l.emit(i, 0, KOp::Embed);
                l.claim(&[i])?;
            }
            "lev::causal_conv1d_state.default" | "lev::ssm_conv_ragged.default" => l.rule_conv(i)?,
            _ => {}
        }
    }
    for i in 0..g.nodes.len() {
        if l.claimed[i] {
            continue;
        }
        match g.nodes[i].op.as_str() {
            "lev::rmsnorm.default" => {
                l.rule_norm_linears(i)?;
            }
            "lev::gated_delta_rule.default" | "lev::gated_delta_ragged.default" => l.rule_delta(i)?,
            "lev::causal_attention.default" => l.rule_attention(i)?,
            "lev::slab_attention.default" => l.rule_slab_attention(i)?,
            "lev::paged_attention.default" => l.rule_paged_attention(i)?,
            "lev::swiglu.default" => l.rule_swiglu(i)?,
            _ => {}
        }
    }
    for i in 0..g.nodes.len() {
        if !l.claimed[i] && g.nodes[i].op == "aten::linear.default" {
            l.rule_linear(i)?;
        }
    }
    if let Some(i) = l.claimed.iter().position(|c| !c) {
        return err(format!("node {i} ({}) is not covered by any rewrite rule", g.nodes[i].op));
    }
    let mut ops = l.ops;
    ops.sort_by_key(|(anchor, seq, _)| (*anchor, *seq));
    let mut ops: Vec<KOp> = ops.into_iter().map(|(_, _, o)| o).collect();
    // Greedy token selection after the output projection is the runtime's, not the graph's.
    if !matches!(ops.last(), Some(KOp::Matvec { out: Slot::Logits, .. })) {
        return err("the graph does not end with the output projection");
    }
    ops.push(KOp::Argmax);
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lev_ir::pass::retarget_gguf;

    fn tiny() -> Graph {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../lev-ir/tests/fixtures/tiny/ir.json");
        retarget_gguf(&Graph::parse(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap()
    }

    fn count(ops: &[KOp], f: impl Fn(&KOp) -> bool) -> usize {
        ops.iter().filter(|o| f(o)).count()
    }

    #[test]
    fn lowers_the_tiny_model_completely() {
        let ops = lower(&tiny()).unwrap();
        // 3 gated delta layers + 1 attention layer, each with the FFN.
        assert_eq!(ops[0], KOp::Embed);
        assert_eq!(count(&ops, |o| matches!(o, KOp::NormQuant { .. })), 9);
        assert_eq!(count(&ops, |o| matches!(o, KOp::SsmConv { .. })), 3);
        assert_eq!(count(&ops, |o| matches!(o, KOp::SsmDelta { .. })), 3);
        assert_eq!(count(&ops, |o| matches!(o, KOp::AttnPrep { .. })), 1);
        assert_eq!(count(&ops, |o| matches!(o, KOp::SwigluQuant)), 4);
        // gated delta layer: 4 + out + 3 FFN = 8; attention layer: 3 + o + 3 = 7; plus the head.
        assert_eq!(count(&ops, |o| matches!(o, KOp::Matvec { .. })), 3 * 8 + 7 + 1);
        assert_eq!(count(&ops, |o| matches!(o, KOp::Quant { .. })), 4);
        assert_eq!(ops.last(), Some(&KOp::Argmax));
        // The first norm has no residual to fold in, every later norm folds one in.
        let adds: Vec<_> = ops.iter().filter_map(|o| if let KOp::NormQuant { add, .. } = o { Some(add.is_some()) } else { None }).collect();
        assert_eq!(adds[0], false);
        assert!(adds[1..].iter().all(|&a| a));
        // Only the final norm keeps the hidden state.
        assert_eq!(count(&ops, |o| matches!(o, KOp::NormQuant { keep_hidden: true, .. })), 1);
    }

    #[test]
    fn lowers_the_ragged_tiny_model_completely() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../lev-ir/tests/fixtures/tiny_ragged/ir.json");
        let g = retarget_gguf(&Graph::parse(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        assert_eq!(g.layout, "ragged-v1");
        let ops = lower(&g).unwrap();
        assert_eq!(count(&ops, |o| matches!(o, KOp::SsmConvBatch { .. })), 3);
        assert_eq!(count(&ops, |o| matches!(o, KOp::SsmDeltaBatch { .. })), 3);
        assert_eq!(count(&ops, |o| matches!(o, KOp::AttnPrepBatch { .. } | KOp::AttnDecodeBatch { .. })), 2);
        assert_eq!(count(&ops, |o| matches!(o, KOp::Matvec { .. })), 3 * 8 + 7 + 1);
        assert_eq!(ops.len(), lower(&tiny()).unwrap().len());
        let plan = crate::plan::plan(&g).unwrap();
        assert_eq!(plan.kinds.len(), 4);
    }

    #[test]
    fn lowers_the_paged_tiny_model_completely() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../lev-ir/tests/fixtures/tiny_paged/ir.json");
        let g = retarget_gguf(&Graph::parse(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        assert_eq!(g.layout, "paged-v1");
        let ops = lower(&g).unwrap();
        assert_eq!(count(&ops, |o| matches!(o, KOp::AttnPrepPaged { .. } | KOp::AttnDecodePaged { .. })), 2);
        assert_eq!(count(&ops, |o| matches!(o, KOp::AttnPrepBatch { .. } | KOp::AttnDecodeBatch { .. })), 0);
        assert_eq!(ops.len(), lower(&tiny()).unwrap().len());
        assert_eq!(crate::plan::plan(&g).unwrap().kinds.len(), 4);
    }

    #[test]
    fn lowers_the_bundled_ornith_ir_to_the_known_command_stream() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("ir/ornith-1.5-9b.json");
        let g = retarget_gguf(&Graph::parse(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let ops = lower(&g).unwrap();
        // 468 kernel operations = the dispatches of the hand-written decode pass.
        assert_eq!(ops.len(), 468);
        assert_eq!(count(&ops, |o| matches!(o, KOp::NormQuant { .. })), 65);
        assert_eq!(count(&ops, |o| matches!(o, KOp::SsmConv { .. } | KOp::SsmQkNorm | KOp::SsmDelta { .. })), 72);
        assert_eq!(count(&ops, |o| matches!(o, KOp::AttnPrep { .. } | KOp::AttnDecode { .. })), 16);
        assert_eq!(count(&ops, |o| matches!(o, KOp::SwigluQuant)), 32);
        assert_eq!(count(&ops, |o| matches!(o, KOp::Quant { .. })), 32);
        assert_eq!(count(&ops, |o| matches!(o, KOp::Matvec { .. })), 249);
    }
    #[test]
    fn lowers_the_bundled_ragged_ornith_ir() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("ir/ornith-1.5-9b-ragged.json");
        let g = retarget_gguf(&Graph::parse(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        assert_eq!(g.layout, "ragged-v1");
        let ops = lower(&g).unwrap();
        assert_eq!(ops.len(), 468);
        assert_eq!(count(&ops, |o| matches!(o, KOp::SsmConvBatch { .. })), 24);
        assert_eq!(count(&ops, |o| matches!(o, KOp::SsmDeltaBatch { .. })), 24);
        assert_eq!(count(&ops, |o| matches!(o, KOp::AttnPrepBatch { .. } | KOp::AttnDecodeBatch { .. })), 16);
        assert_eq!(count(&ops, |o| matches!(o, KOp::Matvec { .. })), 249);
        // No operation of the single-sequence layout remains.
        assert_eq!(count(&ops, |o| matches!(o, KOp::SsmConv { .. } | KOp::SsmDelta { .. } | KOp::AttnPrep { .. } | KOp::AttnDecode { .. })), 0);
    }
    #[test]
    fn rejects_a_graph_with_an_operator_no_rule_covers() {
        let mut g = tiny();
        let (first, end) = g.layer_segment(0).unwrap();
        let i = (first..end).find(|&i| g.nodes[i].op == "aten::sigmoid.default").unwrap();
        g.nodes[i].op = "aten::exp.default".into();
        assert!(lower(&g).is_err());
    }
}
