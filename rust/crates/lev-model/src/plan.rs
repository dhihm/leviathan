//! Execution plan extracted from the model IR: how many layers, of which kind, and which GGUF
//! tensors they read. The GPU program is built from kernels that implement the operator sequences
//! of `SSM_TEMPLATE` and `ATTN_TEMPLATE`; an IR whose layers differ from them is rejected, so the
//! hand-written emitters can never silently run a different graph than the one that was exported.

use lev_ir::bind;
use lev_ir::ir::Graph;

use crate::{Config, Error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated delta net (linear attention) layer followed by the SwiGLU FFN.
    Ssm,
    /// Full attention layer followed by the SwiGLU FFN.
    Attn,
}

/// Operator sequence of a gated delta net layer in the retargeted IR (`pass::retarget_gguf`).
pub const SSM_TEMPLATE: &[&str] = &[
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::linear.default",
    "aten::linear.default",
    "aten::linear.default",
    "aten::squeeze.dim",
    "lev::causal_conv1d_state.default",
    "getitem",
    "getitem",
    "aten::split_with_sizes.default",
    "getitem",
    "getitem",
    "getitem",
    "aten::reshape.default",
    "aten::reshape.default",
    "aten::reshape.default",
    "aten::sigmoid.default",
    "aten::to.dtype",
    "aten::add.Tensor",
    "aten::softplus.default",
    "aten::mul.Tensor",
    "lev::tile_heads.default",
    "lev::tile_heads.default",
    "lev::gated_delta_rule.default",
    "getitem",
    "getitem",
    "aten::reshape.default",
    "aten::reshape.default",
    "lev::rmsnorm_gated.default",
    "aten::reshape.default",
    "aten::linear.default",
    "aten::add.Tensor",
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::linear.default",
    "lev::swiglu.default",
    "aten::linear.default",
    "aten::add.Tensor",
];

/// Operator sequence of a full attention layer in the retargeted IR.
pub const ATTN_TEMPLATE: &[&str] = &[
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::view.default",
    "aten::chunk.default",
    "getitem",
    "getitem",
    "aten::reshape.default",
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::view.default",
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::view.default",
    "lev::rope.default",
    "lev::rope.default",
    "aten::cat.default",
    "aten::cat.default",
    "lev::causal_attention.default",
    "aten::reshape.default",
    "aten::sigmoid.default",
    "aten::mul.Tensor",
    "aten::linear.default",
    "aten::add.Tensor",
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::linear.default",
    "lev::swiglu.default",
    "aten::linear.default",
    "aten::add.Tensor",
];

/// Operator sequences of the ragged (multi-sequence) layout: state pools, per-token slot and position.
pub const SSM_RAGGED_TEMPLATE: &[&str] = &[
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::linear.default",
    "aten::linear.default",
    "aten::linear.default",
    "aten::squeeze.dim",
    "lev::ssm_conv_ragged.default",
    "getitem",
    "getitem",
    "aten::split_with_sizes.default",
    "getitem",
    "getitem",
    "getitem",
    "aten::reshape.default",
    "aten::reshape.default",
    "aten::reshape.default",
    "aten::sigmoid.default",
    "aten::to.dtype",
    "aten::add.Tensor",
    "aten::softplus.default",
    "aten::mul.Tensor",
    "lev::tile_heads.default",
    "lev::tile_heads.default",
    "lev::gated_delta_ragged.default",
    "getitem",
    "getitem",
    "aten::reshape.default",
    "aten::reshape.default",
    "lev::rmsnorm_gated.default",
    "aten::reshape.default",
    "aten::linear.default",
    "aten::add.Tensor",
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::linear.default",
    "lev::swiglu.default",
    "aten::linear.default",
    "aten::add.Tensor",
];

pub const ATTN_RAGGED_TEMPLATE: &[&str] = &[
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::view.default",
    "aten::chunk.default",
    "getitem",
    "getitem",
    "aten::reshape.default",
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::view.default",
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::view.default",
    "lev::rope.default",
    "lev::rope.default",
    "lev::slab_kv_write.default",
    "getitem",
    "getitem",
    "lev::slab_attention.default",
    "aten::reshape.default",
    "aten::sigmoid.default",
    "aten::mul.Tensor",
    "aten::linear.default",
    "aten::add.Tensor",
    "lev::rmsnorm.default",
    "aten::linear.default",
    "aten::linear.default",
    "lev::swiglu.default",
    "aten::linear.default",
    "aten::add.Tensor",
];

#[derive(Debug, Clone)]
pub struct IrPlan {
    pub kinds: Vec<LayerKind>,
    /// GGUF tensors read by the trunk, in graph order, each once.
    pub tensors: Vec<String>,
}

/// Extracts the plan from a graph that went through `pass::retarget_gguf`.
pub fn plan(g: &Graph) -> Result<IrPlan, Error> {
    let n = g.layer_count();
    if n == 0 {
        return Err("the graph has no decoder layers".into());
    }
    let mut kinds = Vec::with_capacity(n);
    for l in 0..n {
        let (first, end) = g.layer_segment(l).ok_or_else(|| format!("layer {l}: no nodes"))?;
        let ops: Vec<&str> = g.nodes[first..end].iter().map(|x| if x.op.ends_with("getitem") { "getitem" } else { x.op.as_str() }).collect();
        let ragged = g.layout == "ragged-v1";
        let ssm = ops.contains(&"lev::gated_delta_rule.default") || ops.contains(&"lev::gated_delta_ragged.default");
        let (kind, template) = match (ssm, ragged) {
            (true, false) => (LayerKind::Ssm, SSM_TEMPLATE),
            (false, false) => (LayerKind::Attn, ATTN_TEMPLATE),
            (true, true) => (LayerKind::Ssm, SSM_RAGGED_TEMPLATE),
            (false, true) => (LayerKind::Attn, ATTN_RAGGED_TEMPLATE),
        };
        if ops != template {
            let at = ops.iter().zip(template).position(|(a, b)| a != b).unwrap_or(ops.len().min(template.len()));
            return Err(format!(
                "layer {l} ({kind:?}) does not match the operator sequence the kernels implement: node {at} is {:?}, expected {:?} ({} nodes, expected {})",
                ops.get(at),
                template.get(at),
                ops.len(),
                template.len()
            )
            .into());
        }
        kinds.push(kind);
    }
    let mut tensors = Vec::new();
    for p in &g.params {
        let (name, _) = bind::to_gguf(&p.name).ok_or_else(|| format!("no GGUF binding for {}", p.name))?;
        if !tensors.contains(&name) {
            tensors.push(name);
        }
    }
    Ok(IrPlan { kinds, tensors })
}

impl IrPlan {
    /// The plan must describe the model the GGUF file holds.
    pub fn check(&self, cfg: &Config) -> Result<(), Error> {
        if self.kinds.len() != cfg.n_trunk {
            return Err(format!("the IR has {} layers, the model file {}", self.kinds.len(), cfg.n_trunk).into());
        }
        for (il, k) in self.kinds.iter().enumerate() {
            if (*k == LayerKind::Ssm) != cfg.is_recurrent(il) {
                return Err(format!("layer {il}: the IR says {k:?}, the model file says otherwise").into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lev_ir::pass::retarget_gguf;

    fn tiny() -> Graph {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../lev-ir/tests/fixtures/tiny/ir.json");
        retarget_gguf(&Graph::parse(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap()
    }

    #[test]
    fn extracts_layer_kinds_and_tensors() {
        let p = plan(&tiny()).unwrap();
        assert_eq!(p.kinds, [LayerKind::Ssm, LayerKind::Ssm, LayerKind::Ssm, LayerKind::Attn]);
        assert!(p.tensors.contains(&"blk.3.attn_q.weight".to_string()));
        assert!(p.tensors.contains(&"blk.0.ssm_a".to_string()));
        assert!(p.tensors.contains(&"token_embd.weight".to_string()));
        assert!(!p.tensors.iter().any(|t| t.contains("A_log")));
    }

    #[test]
    fn rejects_a_layer_that_differs_from_the_kernel_template() {
        let mut g = tiny();
        let (first, end) = g.layer_segment(1).unwrap();
        let i = (first..end).find(|&i| g.nodes[i].op == "lev::swiglu.default").unwrap();
        g.nodes[i].op = "aten::mul.Tensor".into();
        let e = plan(&g).unwrap_err().to_string();
        assert!(e.contains("layer 1") && e.contains("lev::swiglu.default"), "{e}");
    }
}