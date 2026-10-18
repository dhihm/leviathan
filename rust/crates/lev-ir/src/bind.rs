//! Binding of IR parameters (Hugging Face names) to GGUF tensors.
//!
//! The IR carries no weights. A parameter is bound to the GGUF tensor `llama.cpp` writes for it;
//! a few tensors are stored in a transformed form, which the lowering must account for:
//!
//! * RMSNorm weights: Hugging Face scales by `1 + w`, GGUF stores `1 + w`. An IR `rmsnorm` with
//!   `zero_centered = true` therefore runs on GGUF weights as `zero_centered = false`.
//! * `A_log` (the decay base): GGUF stores `ssm_a = -exp(A_log)`; the IR graph computes that from
//!   `A_log` with `to`, `exp`, `neg`, which constant folding replaces by the stored tensor.
//! * Value-head order: the converter reorders the value heads of the linear-attention layers
//!   (Hugging Face groups them with `repeat_interleave`, GGUF tiles them). Not visible in names or
//!   shapes; checked numerically against the CPU reference.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Same values and layout (2-D weights have their two dimensions reversed in GGUF).
    Plain,
    /// Norm weight: GGUF holds `1 + w` where Hugging Face holds `w`.
    NormPlusOne,
    /// GGUF holds `-exp(A_log)`.
    NegExp,
}

/// GGUF tensor name and transform for a Hugging Face parameter name, or `None` if unknown.
pub fn to_gguf(name: &str) -> Option<(String, Transform)> {
    use Transform::*;
    match name {
        "model.embed_tokens.weight" => return Some(("token_embd.weight".into(), Plain)),
        "model.norm.weight" => return Some(("output_norm.weight".into(), NormPlusOne)),
        "lm_head.weight" => return Some(("output.weight".into(), Plain)),
        _ => {}
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, tail) = rest.split_once('.')?;
    layer.parse::<usize>().ok()?;
    let (suffix, t) = match tail {
        "input_layernorm.weight" => ("attn_norm.weight", NormPlusOne),
        "post_attention_layernorm.weight" => ("post_attention_norm.weight", NormPlusOne),
        "mlp.gate_proj.weight" => ("ffn_gate.weight", Plain),
        "mlp.up_proj.weight" => ("ffn_up.weight", Plain),
        "mlp.down_proj.weight" => ("ffn_down.weight", Plain),
        "linear_attn.in_proj_qkv.weight" => ("attn_qkv.weight", Plain),
        "linear_attn.in_proj_z.weight" => ("attn_gate.weight", Plain),
        "linear_attn.in_proj_b.weight" => ("ssm_beta.weight", Plain),
        "linear_attn.in_proj_a.weight" => ("ssm_alpha.weight", Plain),
        "linear_attn.conv1d.weight" => ("ssm_conv1d.weight", Plain),
        "linear_attn.A_log" => ("ssm_a", NegExp),
        // After pass::retarget_gguf: the folded decay base is the stored tensor itself.
        "linear_attn.ssm_a" => ("ssm_a", Plain),
        "linear_attn.dt_bias" => ("ssm_dt.bias", Plain),
        "linear_attn.norm.weight" => ("ssm_norm.weight", Plain),
        "linear_attn.out_proj.weight" => ("ssm_out.weight", Plain),
        "self_attn.q_proj.weight" => ("attn_q.weight", Plain),
        "self_attn.k_proj.weight" => ("attn_k.weight", Plain),
        "self_attn.v_proj.weight" => ("attn_v.weight", Plain),
        "self_attn.o_proj.weight" => ("attn_output.weight", Plain),
        "self_attn.q_norm.weight" => ("attn_q_norm.weight", NormPlusOne),
        "self_attn.k_norm.weight" => ("attn_k_norm.weight", NormPlusOne),
        _ => return None,
    };
    Some((format!("blk.{layer}.{suffix}"), t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_known_names() {
        assert_eq!(to_gguf("model.layers.3.self_attn.q_proj.weight"), Some(("blk.3.attn_q.weight".into(), Transform::Plain)));
        assert_eq!(to_gguf("model.layers.0.linear_attn.A_log"), Some(("blk.0.ssm_a".into(), Transform::NegExp)));
        assert_eq!(to_gguf("model.layers.12.input_layernorm.weight"), Some(("blk.12.attn_norm.weight".into(), Transform::NormPlusOne)));
        assert_eq!(to_gguf("model.embed_tokens.weight").unwrap().0, "token_embd.weight");
        assert_eq!(to_gguf("model.layers.0.unknown.weight"), None);
        assert_eq!(to_gguf("something.else"), None);
    }
}
