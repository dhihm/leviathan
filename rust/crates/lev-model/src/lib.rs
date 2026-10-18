//! Ornith-1.5 (`qwen35`) CPU reference forward pass.
//!
//! One token at a time, f32, weights streamed from the GGUF file per use. It
//! is slow on purpose: its job is to be the layer-by-layer oracle for the GPU
//! path and to be checked against llama.cpp logits. Structure follows the
//! documented qwen35 graph: hybrid layers (gated delta net, full attention
//! every 4th layer), each followed by a SwiGLU FFN.

use std::fs::File;

use lev_cpu::{matvec_q4k_f32, matvec_q6k_f32, QK_K};
use lev_gguf::{GgmlType, GgufFile};

pub mod gpu;
pub mod plan;
pub mod weights;

pub type Error = Box<dyn std::error::Error>;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_embd: usize,
    pub n_trunk: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rope_dim: usize,
    pub rope_base: f64,
    pub eps: f32,
    pub interval: usize,
    pub conv_k: usize,
    pub k_heads: usize,
    pub inner: usize,
    pub state: usize,
    pub v_heads: usize,
    pub vocab: usize,
}

impl Config {
    pub fn from_gguf(g: &GgufFile) -> Result<Config, Error> {
        let u = |k: &str| g.meta_u64(&format!("qwen35.{k}")).ok_or_else(|| format!("missing metadata qwen35.{k}"));
        let eps = match g.metadata.get("qwen35.attention.layer_norm_rms_epsilon") {
            Some(lev_gguf::Value::F32(v)) => *v,
            _ => return Err("missing rms epsilon".into()),
        };
        let rope_base = match g.metadata.get("qwen35.rope.freq_base") {
            Some(lev_gguf::Value::F32(v)) => *v as f64,
            Some(v) => v.as_u64().ok_or("rope base")? as f64,
            None => return Err("missing rope base".into()),
        };
        let vocab = g.tensor("token_embd.weight").ok_or("token_embd.weight")?.dims[1] as usize;
        let c = Config {
            n_embd: u("embedding_length")? as usize,
            n_trunk: (u("block_count")? - u("nextn_predict_layers").unwrap_or(0)) as usize,
            n_head: u("attention.head_count")? as usize,
            n_kv: u("attention.head_count_kv")? as usize,
            head_dim: u("attention.key_length")? as usize,
            n_ff: u("feed_forward_length")? as usize,
            rope_dim: u("rope.dimension_count")? as usize,
            rope_base,
            eps,
            interval: u("full_attention_interval")? as usize,
            conv_k: u("ssm.conv_kernel")? as usize,
            k_heads: u("ssm.group_count")? as usize,
            inner: u("ssm.inner_size")? as usize,
            state: u("ssm.state_size")? as usize,
            v_heads: u("ssm.time_step_rank")? as usize,
            vocab,
        };
        if c.inner != c.state * c.v_heads || c.v_heads % c.k_heads != 0 {
            return Err("unsupported ssm head configuration".into());
        }
        Ok(c)
    }

    pub fn is_recurrent(&self, il: usize) -> bool {
        (il + 1) % self.interval != 0
    }

    pub fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.state + self.inner
    }
}

enum LayerState {
    Ssm { conv: Vec<f32>, s: Vec<f32> },
    Attn { k: Vec<f32>, v: Vec<f32> },
}

pub struct Model {
    pub cfg: Config,
    gguf: GgufFile,
    file: File,
    layers: Vec<LayerState>,
    pos: usize,
    /// Quantize every matvec input to int8 (groups of 32) like the GPU path.
    pub q8_activations: bool,
}

/// Per-token intermediate values for comparing another implementation.
#[derive(Default)]
pub struct Trace {
    /// Residual stream after each trunk layer.
    pub layer_out: Vec<Vec<f32>>,
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps as f64).sqrt();
    x.iter().zip(w).map(|(v, w)| (*v as f64 * inv) as f32 * w).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

impl Model {
    pub fn open(path: &str) -> Result<Model, Error> {
        let gguf = GgufFile::open(path)?;
        let cfg = Config::from_gguf(&gguf)?;
        let file = File::open(path)?;
        let layers = (0..cfg.n_trunk)
            .map(|il| {
                if cfg.is_recurrent(il) {
                    LayerState::Ssm {
                        conv: vec![0.0; (cfg.conv_k - 1) * cfg.conv_dim()],
                        s: vec![0.0; cfg.v_heads * cfg.state * cfg.state],
                    }
                } else {
                    LayerState::Attn { k: Vec::new(), v: Vec::new() }
                }
            })
            .collect();
        Ok(Model { cfg, gguf, file, layers, pos: 0, q8_activations: false })
    }

    fn f32_tensor(&mut self, name: &str) -> Result<Vec<f32>, Error> {
        let t = self.gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        if t.ty != GgmlType::F32 {
            return Err(format!("{name}: expected F32").into());
        }
        let n = t.byte_size().ok_or("size")?;
        let bytes = self.gguf.read_tensor_bytes(&mut self.file, t, 0, n)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// `y = W x` for a 2-D Q4_K or Q6_K tensor, streamed in row chunks and
    /// computed on all cores.
    fn matvec(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>, Error> {
        let t = self.gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        if cols != x.len() {
            return Err(format!("{name}: input length {} != cols {cols}", x.len()).into());
        }
        let xq: Vec<f32>;
        let x: &[f32] = if self.q8_activations {
            let q = lev_cpu::quantize_q8(x);
            xq = q.q.iter().enumerate().map(|(i, &v)| v as f32 * q.d[i / 32]).collect();
            &xq
        } else {
            x
        };
        let (block_bytes, f): (usize, fn(&[u8], usize, usize, &[f32]) -> Result<Vec<f32>, lev_cpu::ShapeError>) = match t.ty {
            GgmlType::Q4K => (lev_cpu::Q4K_BLOCK_BYTES, matvec_q4k_f32),
            GgmlType::Q6K => (lev_cpu::Q6K_BLOCK_BYTES, matvec_q6k_f32),
            other => return Err(format!("{name}: unsupported type {}", other.name()).into()),
        };
        let row_bytes = cols / QK_K * block_bytes;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk_rows = (64usize << 20).div_ceil(row_bytes).max(threads);
        let mut y = Vec::with_capacity(rows);
        let mut r0 = 0;
        while r0 < rows {
            let n = chunk_rows.min(rows - r0);
            let t = self.gguf.tensor(name).unwrap();
            let bytes = self.gguf.read_tensor_bytes(&mut self.file, t, (r0 * row_bytes) as u64, (n * row_bytes) as u64)?;
            let per = n.div_ceil(threads);
            let parts: Vec<Result<Vec<f32>, lev_cpu::ShapeError>> = std::thread::scope(|s| {
                let handles: Vec<_> = bytes
                    .chunks(per * row_bytes)
                    .map(|part| s.spawn(move || f(part, part.len() / row_bytes, cols, x)))
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for p in parts {
                y.extend(p?);
            }
            r0 += n;
        }
        Ok(y)
    }

    fn embed(&mut self, token: usize) -> Result<Vec<f32>, Error> {
        let t = self.gguf.tensor("token_embd.weight").ok_or("token_embd.weight")?;
        let cols = t.dims[0] as usize;
        let row_bytes = cols / QK_K * lev_cpu::Q4K_BLOCK_BYTES;
        if token >= t.dims[1] as usize {
            return Err("token id out of range".into());
        }
        let bytes = self.gguf.read_tensor_bytes(&mut self.file, t, (token * row_bytes) as u64, row_bytes as u64)?;
        let mut out = Vec::with_capacity(cols);
        let mut tmp = [0f32; QK_K];
        for block in bytes.chunks_exact(lev_cpu::Q4K_BLOCK_BYTES) {
            lev_cpu::dequant_block_f32(block, &mut tmp);
            out.extend_from_slice(&tmp);
        }
        Ok(out)
    }

    /// Runs one token at the next position and returns the logits.
    pub fn forward(&mut self, token: usize, mut trace: Option<&mut Trace>) -> Result<Vec<f32>, Error> {
        let c = self.cfg.clone();
        let mut x = self.embed(token)?;
        for il in 0..c.n_trunk {
            let norm = self.f32_tensor(&format!("blk.{il}.attn_norm.weight"))?;
            let xn = rmsnorm(&x, &norm, c.eps);
            let mixed = if c.is_recurrent(il) { self.ssm_layer(il, &xn)? } else { self.attn_layer(il, &xn)? };
            for (a, b) in x.iter_mut().zip(&mixed) {
                *a += b;
            }
            let pn = self.f32_tensor(&format!("blk.{il}.post_attention_norm.weight"))?;
            let hn = rmsnorm(&x, &pn, c.eps);
            let gate = self.matvec(&format!("blk.{il}.ffn_gate.weight"), &hn)?;
            let up = self.matvec(&format!("blk.{il}.ffn_up.weight"), &hn)?;
            let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let down = self.matvec(&format!("blk.{il}.ffn_down.weight"), &act)?;
            for (a, b) in x.iter_mut().zip(&down) {
                *a += b;
            }
            if let Some(t) = trace.as_deref_mut() {
                t.layer_out.push(x.clone());
            }
        }
        self.pos += 1;
        let on = self.f32_tensor("output_norm.weight")?;
        let h = rmsnorm(&x, &on, c.eps);
        self.matvec("output.weight", &h)
    }

    fn attn_layer(&mut self, il: usize, xn: &[f32]) -> Result<Vec<f32>, Error> {
        let c = self.cfg.clone();
        let hd = c.head_dim;
        let qfull = self.matvec(&format!("blk.{il}.attn_q.weight"), xn)?;
        let mut k = self.matvec(&format!("blk.{il}.attn_k.weight"), xn)?;
        let v = self.matvec(&format!("blk.{il}.attn_v.weight"), xn)?;
        let qn = self.f32_tensor(&format!("blk.{il}.attn_q_norm.weight"))?;
        let kn = self.f32_tensor(&format!("blk.{il}.attn_k_norm.weight"))?;
        // Q and gate are interleaved per head: [q(hd) | gate(hd)] x n_head.
        let mut q = vec![0f32; c.n_head * hd];
        let mut gate = vec![0f32; c.n_head * hd];
        for h in 0..c.n_head {
            let src = &qfull[h * 2 * hd..(h + 1) * 2 * hd];
            q[h * hd..(h + 1) * hd].copy_from_slice(&rmsnorm(&src[..hd], &qn, c.eps));
            gate[h * hd..(h + 1) * hd].copy_from_slice(&src[hd..]);
        }
        for h in 0..c.n_kv {
            let n = rmsnorm(&k[h * hd..(h + 1) * hd], &kn, c.eps);
            k[h * hd..(h + 1) * hd].copy_from_slice(&n);
        }
        let pos = self.pos as f64;
        let half = c.rope_dim / 2;
        let rope = |v: &mut [f32]| {
            for i in 0..half {
                let theta = pos * c.rope_base.powf(-2.0 * i as f64 / c.rope_dim as f64);
                let (s, co) = theta.sin_cos();
                let (a, b) = (v[i] as f64, v[i + half] as f64);
                v[i] = (a * co - b * s) as f32;
                v[i + half] = (a * s + b * co) as f32;
            }
        };
        for h in 0..c.n_head {
            rope(&mut q[h * hd..(h + 1) * hd]);
        }
        for h in 0..c.n_kv {
            rope(&mut k[h * hd..(h + 1) * hd]);
        }
        let LayerState::Attn { k: kc, v: vc } = &mut self.layers[il] else { return Err("layer state mismatch".into()) };
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let n_pos = kc.len() / (c.n_kv * hd);
        let group = c.n_head / c.n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; c.n_head * hd];
        for h in 0..c.n_head {
            let kvh = h / group;
            let qh = &q[h * hd..(h + 1) * hd];
            let mut scores: Vec<f32> = (0..n_pos)
                .map(|t| {
                    let kt = &kc[(t * c.n_kv + kvh) * hd..(t * c.n_kv + kvh + 1) * hd];
                    qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                sum += *s;
            }
            for (t, p) in scores.iter().enumerate() {
                let vt = &vc[(t * c.n_kv + kvh) * hd..(t * c.n_kv + kvh + 1) * hd];
                for d in 0..hd {
                    out[h * hd + d] += p / sum * vt[d];
                }
            }
        }
        for (o, g) in out.iter_mut().zip(&gate) {
            *o *= sigmoid(*g);
        }
        self.matvec(&format!("blk.{il}.attn_output.weight"), &out)
    }

    fn ssm_layer(&mut self, il: usize, xn: &[f32]) -> Result<Vec<f32>, Error> {
        let c = self.cfg.clone();
        let (hk, nk, nv) = (c.state, c.k_heads, c.v_heads);
        let key_dim = hk * nk;
        let cd = c.conv_dim();
        let qkv = self.matvec(&format!("blk.{il}.attn_qkv.weight"), xn)?;
        let z = self.matvec(&format!("blk.{il}.attn_gate.weight"), xn)?;
        let beta: Vec<f32> = self.matvec(&format!("blk.{il}.ssm_beta.weight"), xn)?.into_iter().map(sigmoid).collect();
        let alpha = self.matvec(&format!("blk.{il}.ssm_alpha.weight"), xn)?;
        let dt = self.f32_tensor(&format!("blk.{il}.ssm_dt.bias"))?;
        let a = self.f32_tensor(&format!("blk.{il}.ssm_a"))?;
        let w = self.f32_tensor(&format!("blk.{il}.ssm_conv1d.weight"))?;
        let gnorm = self.f32_tensor(&format!("blk.{il}.ssm_norm.weight"))?;
        let g: Vec<f32> = (0..nv).map(|h| softplus(alpha[h] + dt[h]) * a[h]).collect();

        let LayerState::Ssm { conv, s } = &mut self.layers[il] else { return Err("layer state mismatch".into()) };
        // Causal conv over [3 previous inputs, current input], tap i weights window position i.
        let mut conv_out = vec![0f32; cd];
        for ch in 0..cd {
            let mut acc = 0f32;
            for i in 0..c.conv_k - 1 {
                acc += conv[i * cd + ch] * w[ch * c.conv_k + i];
            }
            acc += qkv[ch] * w[ch * c.conv_k + c.conv_k - 1];
            conv_out[ch] = silu(acc);
        }
        conv.copy_within(cd.., 0);
        conv[(c.conv_k - 2) * cd..].copy_from_slice(&qkv);

        let l2 = |v: &[f32]| -> Vec<f32> {
            let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt().max(1e-6);
            v.iter().map(|x| (*x as f64 / n) as f32).collect()
        };
        let qscale = 1.0 / (hk as f32).sqrt();
        let mut out = vec![0f32; c.inner];
        for h in 0..nv {
            // Key heads are tiled over value heads (ggml repeat), not interleaved.
            let kh = h % nk;
            let q: Vec<f32> = l2(&conv_out[kh * hk..(kh + 1) * hk]).iter().map(|v| v * qscale).collect();
            let k = l2(&conv_out[key_dim + kh * hk..key_dim + (kh + 1) * hk]);
            let v = &conv_out[2 * key_dim + h * hk..2 * key_dim + (h + 1) * hk];
            let decay = g[h].exp();
            let st = &mut s[h * hk * hk..(h + 1) * hk * hk];
            let mut o = vec![0f32; hk];
            for j in 0..hk {
                let row = &mut st[j * hk..(j + 1) * hk];
                let mut kv = 0f32;
                for i in 0..hk {
                    row[i] *= decay;
                    kv += row[i] * k[i];
                }
                let delta = beta[h] * (v[j] - kv);
                let mut acc = 0f32;
                for i in 0..hk {
                    row[i] += k[i] * delta;
                    acc += row[i] * q[i];
                }
                o[j] = acc;
            }
            let on = rmsnorm(&o, &gnorm, c.eps);
            for j in 0..hk {
                out[h * hk + j] = on[j] * silu(z[h * hk + j]);
            }
        }
        self.matvec(&format!("blk.{il}.ssm_out.weight"), &out)
    }
}

pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best
}

/// Logit gap between the best and second-best token.
pub fn top_margin(logits: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in logits {
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    a - b
}

pub fn log_softmax_top(logits: &[f32], k: usize) -> Vec<(usize, f32, f32)> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m as f64 + logits.iter().map(|v| ((*v - m) as f64).exp()).sum::<f64>().ln();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
    idx.into_iter().take(k).map(|i| (i, logits[i], (logits[i] as f64 - lse) as f32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_of_constant_vector_is_weight() {
        let y = rmsnorm(&[2.0; 8], &[0.5; 8], 1e-6);
        assert!(y.iter().all(|v| (v - 0.5).abs() < 1e-5));
    }

    #[test]
    fn softplus_matches_definition_and_saturates() {
        assert!((softplus(0.0) - 2f32.ln()).abs() < 1e-6);
        assert_eq!(softplus(30.0), 30.0);
    }

    #[test]
    fn top_logprobs_sum_to_one_over_vocab() {
        let l = [1.0f32, 2.0, 3.0];
        let t = log_softmax_top(&l, 3);
        assert_eq!(t[0].0, 2);
        let p: f32 = t.iter().map(|x| x.2.exp()).sum();
        assert!((p - 1.0).abs() < 1e-5);
    }
}
