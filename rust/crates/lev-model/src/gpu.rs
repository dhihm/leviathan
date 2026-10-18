//! GPU decode path for Ornith (`qwen35`). The single-token and two-token
//! forward passes and the MTP draft head are each recorded once as a Vulkan
//! program and replayed. Weights stay in their GGUF block formats; activations
//! are f32. All programs share the same buffers.
//!
//! Speculative decoding (greedy): the MTP head drafts the token after the next
//! one; the main model then verifies `[next, draft]` in one two-column pass
//! that reads every weight once. Recurrent state is double-buffered (see
//! `lev_forge::ops`) so a rejected draft needs no state copy.

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use lev_cpu::QK_K;
use crate::lower::{KOp, Slot};
use lev_forge::q8::{self, Q8Schedule};
use lev_forge::ops;
use lev_gguf::{GgmlType, GgufFile};
use lev_vulkan::{compile_glsl_cached, Buf, DeviceSelector, Kernel, Program, Step, VulkanContext};

use crate::{Config, Error};

/// Columns per prefill chunk; also the widest batch any program uses.
const MAX_COLS: usize = ops::CHUNK_COLS;
/// Tokens per matrix-core prefill chunk (needs cooperative-matrix support).
const BIG_COLS: usize = lev_forge::coopmat::GEMM_TOKENS;
/// Matrix-core chunk sizes (a 128-token chunk multiplies twice the tokens per
/// dequantized weight); activation buffers are sized for the largest.
const BIG_SIZES: [usize; 2] = [BIG_COLS, 2 * BIG_COLS];
const BIG_MAX: usize = 2 * BIG_COLS;
/// Columns the activation buffers are sized for (the per-column stride of any buffer is its size divided by this).
pub const DEBUG_COLS: usize = BIG_MAX;
fn bidx(nb: usize) -> usize {
    BIG_SIZES.iter().position(|&s| s == nb).expect("unsupported matrix-core chunk size")
}
/// Subgroup size of the matrix-core GEMMs (RDNA runs the WMMA unit natively at 32;
/// the blocked GEMMs are written for 32 lanes, workgroup 128).
const GEMM_WAVE: usize = 32;
/// The MTP head scores only the first MTP_VOCAB tokens (frequent ones in the
/// BPE order): a draft outside it is simply a rejected draft, never a wrong
/// output, and the head's output projection is the largest cost of a draft.
const DEFAULT_MTP_VOCAB: usize = 40960;

fn mtp_vocab(cfg: &Config) -> usize {
    std::env::var("LEV_MTP_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MTP_VOCAB).clamp(1024, cfg.vocab) / 256 * 256
}

/// Int8-activation matvec schedules (chosen from the `lev vk-q8` sweeps).
fn q4_schedule(nb: u32) -> Q8Schedule {
    if std::env::var_os("LEV_OLD_SCHED").is_some() && nb != 8 {
        return Q8Schedule { t: 32, r: 1, sg: 0, nb };
    }
    match nb {
        8 | 3 | 4 | 16 => Q8Schedule { t: 4, r: 16, sg: 1, nb },
        _ => Q8Schedule { t: 64, r: 4, sg: 1, nb },
    }
}

fn q6_schedule(nb: u32) -> Q8Schedule {
    if std::env::var_os("LEV_OLD_SCHED").is_some() && nb != 8 {
        return Q8Schedule { t: 32, r: 4, sg: 1, nb };
    }
    match nb {
        8 | 16 => Q8Schedule { t: 64, r: 1, sg: 1, nb },
        _ => Q8Schedule { t: 32, r: 1, sg: 0, nb },
    }
}

/// Batch sizes of the ragged decode programs (columns of distinct sequences). Up to 8 columns use
/// the int8 matvecs, 64 columns the matrix-core GEMMs (a smaller batch is padded with scratch
/// columns): the int8 matvec stops scaling beyond 8 columns.
const BATCH_SIZES: [usize; 5] = [1, 2, 4, 8, 64];
const BATCH_MAX: usize = 64;

fn bsidx(nb: usize) -> usize {
    BATCH_SIZES.iter().position(|&s| s == nb).expect("unsupported batch size")
}

/// Index of the matvec kernel variant for `nb` columns (1, 2, 8, 3, 4 or 16).
fn kidx(nb: usize) -> usize {
    match nb {
        1 => 0,
        2 => 1,
        8 => 2,
        3 => 3,
        4 => 4,
        16 => 5,
        _ => panic!("unsupported column count {nb}"),
    }
}
struct Kernels {
    norm: Kernel,
    add: Kernel,
    swiglu: Kernel,
    copy: Kernel,
    qknorm: Kernel,
    prep: Kernel,
    attn: Kernel,
    quant: Kernel,
    norm_q: Kernel,
    add_norm_q: Kernel,
    add_norm_q_y: Kernel,
    swiglu_q: Kernel,
    argmax: Kernel,
    argmax_mtp: Kernel,
    embed: Kernel,
    conv: [Kernel; 3],
    delta: [Kernel; 3],
    conv_chunk: Kernel,
    delta_chunk: Kernel,
    cat_e: Kernel,
    cat_h: Kernel,
    q4: [Kernel; 6],
    q6: [Kernel; 6],
    big: Option<BigKernels>,
    batch: Option<BatchKernels>,
}

/// Kernels of the ragged batch decode programs.
struct BatchKernels {
    conv: [Kernel; 5],
    delta: Kernel,
    prep: Kernel,
    decode: Kernel,
    prep_paged: Kernel,
    decode_paged: Kernel,
}

/// Per-layer pools of the ragged batch: one state per slot (the last slot is scratch for padding).
enum LayerPool {
    Ssm { conv: Buf, rec: Buf },
    Attn { k: Buf, v: Buf },
}

/// Kernels of the matrix-core prefill path.
struct BigKernels {
    act16: Kernel,
    gemm_q4: [Kernel; 2],
    gemm_q6: [Kernel; 2],
    conv: [Kernel; 2],
    delta: [Kernel; 2],
}

enum LayerRes {
    Ssm { c0: Buf, c1: Buf, c2: Buf, s0: Buf, s1: Buf, s2: Buf },
    Attn { kc: Buf, vc: Buf },
}

/// Every device buffer; activations are sized for two columns.
struct Res {
    weights: HashMap<String, Buf>,
    layers: Vec<LayerRes>,
    /// Ragged batch pools (empty unless LEV_BATCH_SLOTS is set).
    pools: Vec<LayerPool>,
    /// Paged KV: block table `[slots + 1][max blocks]`, rewritten by the host every step.
    block_table: Buf,
    mtp_kv: (Buf, Buf),
    x: Buf,
    xn: Buf,
    mixed: Buf,
    gate: Buf,
    up: Buf,
    act: Buf,
    down: Buf,
    qkv: Buf,
    z: Buf,
    beta: Buf,
    alpha: Buf,
    conv_out: Buf,
    qk: Buf,
    gated: Buf,
    qfull: Buf,
    kin: Buf,
    vin: Buf,
    q_out: Buf,
    attn_out: Buf,
    hfin: Buf,
    /// MTP block output of the last draft pass (chained drafting probe).
    hmtp: Buf,
    tok_main: Buf,
    tok_mtp: Buf,
    am_main: Buf,
    mg_main: Buf,
    am_mtp: Buf,
    mg_mtp: Buf,
    xq: Buf,
    xs: Buf,
    xh: Buf,
    logits: Buf,
    ctrl: Buf,
    e_in: Buf,
    cat: Buf,
    mtp_logits: Buf,
    trace: Option<Buf>,
}

/// One token of one sequence in a ragged batch: its slot, the token and its position.
#[derive(Clone, Copy, Debug)]
pub struct BatchItem {
    pub slot: usize,
    pub token: usize,
    pub pos: usize,
}

pub struct GpuModel {
    // Drop order: programs first, the device context last.
    main1: Program,
    main2: Program,
    /// Three-column pass (two chained drafts), only for speculative depth 2.
    main3: Program,
    /// Ragged decode programs by batch size (empty unless LEV_BATCH_SLOTS is set).
    batch_programs: Vec<(usize, Program)>,
    batch_slots: usize,
    /// Paged KV bookkeeping (None with LEV_KV_SLAB or without batch slots).
    kv: Option<crate::kv::BlockManager>,
    prefill: Program,
    mtp_fill_chunk: Program,
    bigs: Vec<(usize, Program, Program)>,
    mtp_draft: [Program; 3],
    mtp_chain: Program,
    mtp_fill: [Program; 3],
    labels1: Vec<String>,
    labels2: Vec<String>,
    labels_prefill: Vec<String>,
    labels_big: Vec<String>,
    res: Res,
    _kernels: Kernels,
    gguf: GgufFile,
    file: File,
    pub cfg: Config,
    ctx_len: usize,
    pos: usize,
    sel: u32,
    pub last_gpu_us: f64,
    last_mtp_us: f64,
    ctx: VulkanContext,
}

fn full(b: &Buf) -> (&Buf, u64, u64) {
    (b, 0, b.size())
}

fn part(b: &Buf, off_floats: usize, floats: usize) -> (&Buf, u64, u64) {
    (b, (off_floats * 4) as u64, (floats * 4) as u64)
}

struct Builder<'a> {
    steps: Vec<Step<'a>>,
    labels: Vec<String>,
    /// Matrix-core mode: quant converts to f16 and matvec is a GEMM.
    gemm: bool,
    /// Use the fused norm+quantize and swiglu+quantize kernels (int8 mode).
    fuse: bool,
}

impl<'a> Builder<'a> {
    fn new() -> Self {
        Builder { steps: Vec::new(), labels: Vec::new(), gemm: false, fuse: std::env::var_os("LEV_NO_FUSE").is_none() }
    }

    fn simple(&mut self, label: &str, kernel: &'a Kernel, bindings: Vec<(&'a Buf, u64, u64)>, groups: u32) {
        self.steps.push(Step { kernel, bindings, push: Vec::new(), groups });
        self.labels.push(label.to_string());
    }

    /// Quantizes `elems` activations of `src` (a multiple of 256) to int8 into
    /// the shared `xq`/`xs` buffers read by the following matvecs.
    fn quant(&mut self, r: &'a Res, k: &'a Kernels, src: &'a Buf, elems: usize) {
        if self.gemm {
            let bk = k.big.as_ref().expect("matrix-core kernels");
            self.simple("act16", &bk.act16, vec![part(src, 0, elems), full(&r.xh)], (elems / 256) as u32);
            return;
        }
        self.simple("quantize", &k.quant, vec![part(src, 0, elems), full(&r.xq), full(&r.xs)], (elems / 256) as u32);
    }

    /// `x += add` (if given), RMS norm with weight `wt`, then the input of the
    /// following matvecs: int8 quantization (fused into one kernel) or f16.
    /// `y` additionally receives the f32 result.
    #[allow(clippy::too_many_arguments)]
    fn norm_quant(&mut self, r: &'a Res, k: &'a Kernels, ne: usize, wt: &'a Buf, nb: usize, add: Option<&'a Buf>, y: Option<&'a Buf>) {
        if self.gemm || !self.fuse || (add.is_none() && y.is_some()) {
            if let Some(d) = add {
                self.simple("add", &k.add, vec![part(&r.x, 0, nb * ne), part(d, 0, nb * ne)], (nb * ne / 256) as u32);
            }
            let out = y.unwrap_or(&r.xn);
            self.simple("rmsnorm", &k.norm, vec![part(&r.x, 0, nb * ne), full(wt), part(out, 0, nb * ne)], nb as u32);
            self.quant(r, k, out, nb * ne);
            return;
        }
        let mut bindings = vec![part(&r.x, 0, nb * ne), full(wt), full(&r.xq), full(&r.xs)];
        let kernel = match (add, y) {
            (None, None) => &k.norm_q,
            (Some(d), None) => {
                bindings.push(part(d, 0, nb * ne));
                &k.add_norm_q
            }
            (Some(d), Some(yb)) => {
                bindings.push(part(d, 0, nb * ne));
                bindings.push(part(yb, 0, nb * ne));
                &k.add_norm_q_y
            }
            (None, Some(_)) => unreachable!(),
        };
        self.simple("norm_quant", kernel, bindings, nb as u32);
    }

    /// `act = silu(gate) * up` and its quantization for the next matvec.
    fn swiglu_quant(&mut self, r: &'a Res, k: &'a Kernels, nb: usize, nff: usize) {
        if self.gemm || !self.fuse {
            self.simple("swiglu", &k.swiglu, vec![part(&r.gate, 0, nb * nff), part(&r.up, 0, nb * nff), part(&r.act, 0, nb * nff)], (nb * nff / 256) as u32);
            self.quant(r, k, &r.act, nb * nff);
        } else {
            self.simple("swiglu_quant", &k.swiglu_q, vec![part(&r.gate, 0, nb * nff), part(&r.up, 0, nb * nff), full(&r.xq), full(&r.xs)], (nb * nff / 256) as u32);
        }
    }

    /// `y[nb][rows] = W x[nb][cols]` for a Q4_K or Q6_K tensor; reads the
    /// activations most recently quantized by `quant`.
    fn matvec(&mut self, g: &GgufFile, r: &'a Res, k: &'a Kernels, name: &str, nb: usize, y: (&'a Buf, usize)) -> Result<(), Error> {
        self.matvec_rows(g, r, k, name, nb, y, usize::MAX)
    }

    /// Like matvec but only the first max_rows rows (a prefix of the weights).
    #[allow(clippy::too_many_arguments)]
    fn matvec_rows(&mut self, g: &GgufFile, r: &'a Res, k: &'a Kernels, name: &str, nb: usize, y: (&'a Buf, usize), max_rows: usize) -> Result<(), Error> {
        let t = g.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let (cols, rows) = (t.dims[0] as u32, (t.dims[1] as usize).min(max_rows) as u32);
        let wb = &r.weights[name];
        let push = [rows.to_le_bytes(), (cols / QK_K as u32).to_le_bytes()].concat();
        if self.gemm {
            let bk = k.big.as_ref().expect("matrix-core kernels");
            assert!(rows % 32 == 0 && BIG_SIZES.contains(&nb), "gemm needs 32-row multiples and a matrix-core chunk size");
            let kernel = match t.ty {
                GgmlType::Q4K => &bk.gemm_q4[bidx(nb)],
                GgmlType::Q6K => &bk.gemm_q6[bidx(nb)],
                other => return Err(format!("{name}: unsupported type {}", other.name()).into()),
            };
            self.steps.push(Step { kernel, bindings: vec![full(wb), full(&r.xh), part(y.0, y.1, nb * rows as usize)], push, groups: rows / 32 });
            self.labels.push(format!("gemm {:.1}MB", wb.size() as f64 / 1e6));
            return Ok(());
        }
        let (kernel, groups, label) = match t.ty {
            GgmlType::Q4K => (&k.q4[kidx(nb)], q8::group_count(&q4_schedule(nb as u32), rows), format!("mv_q4k {:.1}MB", wb.size() as f64 / 1e6)),
            GgmlType::Q6K => (&k.q6[kidx(nb)], q8::group_count(&q6_schedule(nb as u32), rows), format!("mv_q6k {:.1}MB", wb.size() as f64 / 1e6)),
            other => return Err(format!("{name}: unsupported type {}", other.name()).into()),
        };
        self.steps.push(Step {
            kernel,
            bindings: vec![full(wb), full(&r.xq), part(y.0, y.1, nb * rows as usize), full(&r.xs)],
            push,
            groups,
        });
        self.labels.push(label);
        Ok(())
    }
}
fn cache_dir() -> PathBuf {
    std::env::var_os("LEV_CACHE_DIR")
        .map(|p| PathBuf::from(p).join("spv"))
        .unwrap_or_else(|| std::env::temp_dir().join("lev-forge").join("spv"))
}

/// Records the main decode pass for `nb` columns.
/// prefill records a chunk pass: chunk-form SSM kernels (only the final state is written) and no output head.
#[allow(clippy::too_many_arguments)]
fn build_main(ctx: &VulkanContext, g: &GgufFile, cfg: &Config, r: &Res, k: &Kernels, nb: usize, profile: bool, prefill: bool, big: bool) -> Result<(Program, Vec<String>), Error> {
    finish_program(ctx, emit_main(g, cfg, r, k, nb, prefill, big)?, profile)
}

/// Structure of Ornith-1.5-9B exported by tools/model_ir/ir_export.py (no weights).
const BUNDLED_IR: &str = include_str!("../ir/ornith-1.5-9b.json");
/// Same model as a ragged multi-sequence batch (state pools, per-token slot and position).
const BUNDLED_RAGGED_IR: &str = include_str!("../ir/ornith-1.5-9b-ragged.json");
/// Same model over a paged KV cache (block pools, block table, per-token pool position).
const BUNDLED_PAGED_IR: &str = include_str!("../ir/ornith-1.5-9b-paged.json");

/// The model IR the decode passes are lowered from: `LEV_IR=<file>` names another export,
/// `LEV_NO_IR=1` selects the hand-written passes (the regression oracle), otherwise the bundled
/// IR is used when it describes this model file.
type IrLowering = (crate::plan::IrPlan, Vec<KOp>);
/// The ragged-batch IR lowered to kernel operations (always from the bundled file).
fn load_ragged_ir(cfg: &Config, paged: bool) -> Result<Vec<KOp>, Error> {
    let g = lev_ir::pass::retarget_gguf(&lev_ir::ir::Graph::parse(if paged { BUNDLED_PAGED_IR } else { BUNDLED_RAGGED_IR })?)?;
    g.validate()?;
    let plan = crate::plan::plan(&g)?;
    plan.check(cfg)?;
    Ok(crate::lower::lower(&g)?)
}

fn load_ir(cfg: &Config) -> Result<Option<IrLowering>, Error> {
    if std::env::var_os("LEV_NO_IR").is_some() {
        return Ok(None);
    }
    let (text, explicit) = match std::env::var_os("LEV_IR") {
        Some(path) => (std::fs::read_to_string(path)?, true),
        None => (BUNDLED_IR.to_string(), false),
    };
    let lowered = (|| -> Result<IrLowering, Error> {
        let g = lev_ir::pass::retarget_gguf(&lev_ir::ir::Graph::parse(&text)?)?;
        g.validate()?;
        let plan = crate::plan::plan(&g)?;
        plan.check(cfg)?;
        let ops = crate::lower::lower(&g)?;
        Ok((plan, ops))
    })();
    match lowered {
        Ok(v) => Ok(Some(v)),
        Err(e) if explicit => Err(e),
        // The bundled IR describes Ornith-1.5-9B: another model file falls back to the hand-written passes.
        Err(e) => {
            eprintln!("note: the bundled model IR does not fit this model file ({e}); using the hand-written passes");
            Ok(None)
        }
    }
}

/// Builds the program of recorded steps; a failing step is named by its label.
fn finish_program(ctx: &VulkanContext, b: Builder<'_>, profile: bool) -> Result<(Program, Vec<String>), Error> {
    let program = match ctx.build_program(&b.steps, profile) {
        Ok(p) => p,
        Err(e) => {
            // Name the failing step: the error text starts with its index.
            let msg = e.to_string();
            let idx = msg.split_whitespace().nth(1).and_then(|t| t.trim_end_matches(':').parse::<usize>().ok());
            return Err(match idx.and_then(|i| b.labels.get(i)) {
                Some(l) => format!("{msg} ({l})").into(),
                None => msg.into(),
            });
        }
    };
    Ok((program, b.labels))
}

/// The steps of the main decode (or prefill chunk) pass, written out by hand.
#[allow(clippy::too_many_arguments)]
fn emit_main<'a>(g: &GgufFile, cfg: &Config, r: &'a Res, k: &'a Kernels, nb: usize, prefill: bool, big: bool) -> Result<Builder<'a>, Error> {
    let (ne, cd) = (cfg.n_embd, cfg.conv_dim());
    let (kh, vh, hs) = (cfg.k_heads, cfg.v_heads, cfg.state);
    let mut b = Builder::new();
    b.gemm = big;
    let w = |n: &str| -> &Buf { &r.weights[n] };
    b.simple("embed", &k.embed, vec![full(w("token_embd.weight")), part(&r.tok_main, 0, nb), part(&r.x, 0, nb * ne)], (nb * ne / 256) as u32);
    // The FFN's closing residual add is deferred into the next norm kernel
    // (unless a trace needs the residual after every layer).
    let defer = r.trace.is_none();
    let mut pending: Option<&Buf> = None;
    // Debug aid: stop after this many layers so their intermediates stay in the buffers.
    let layer_limit: usize = std::env::var("LEV_MAX_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(cfg.n_trunk).min(cfg.n_trunk);
    for il in 0..layer_limit {
        let p = |n: &str| format!("blk.{il}.{n}");
        b.norm_quant(r, k, ne, w(&p("attn_norm.weight")), nb, pending.take(), None);
        match &r.layers[il] {
            LayerRes::Ssm { c0, c1, c2, s0, s1, s2 } => {
                b.matvec(g, r, k, &p("attn_qkv.weight"), nb, (&r.qkv, 0))?;
                b.matvec(g, r, k, &p("attn_gate.weight"), nb, (&r.z, 0))?;
                b.matvec(g, r, k, &p("ssm_beta.weight"), nb, (&r.beta, 0))?;
                b.matvec(g, r, k, &p("ssm_alpha.weight"), nb, (&r.alpha, 0))?;
                let mut conv_b = vec![part(&r.qkv, 0, nb * cd), full(c0), full(c1), full(w(&p("ssm_conv1d.weight"))), part(&r.conv_out, 0, nb * cd), full(&r.ctrl)];
                if nb == 3 && !prefill {
                    conv_b.push(full(c2));
                }
                b.simple("ssm_conv", if big { &k.big.as_ref().unwrap().conv[bidx(nb)] } else if prefill { &k.conv_chunk } else { &k.conv[nb - 1] }, conv_b, (cd / 256) as u32);
                b.simple("ssm_qknorm", &k.qknorm, vec![part(&r.conv_out, 0, nb * cd), part(&r.qk, 0, nb * 2 * kh * hs)], (nb * 2 * kh) as u32);
                let mut delta_b = 
                    vec![
                        part(&r.qk, 0, nb * 2 * kh * hs),
                        part(&r.conv_out, 0, nb * cd),
                        part(&r.alpha, 0, nb * vh),
                        part(&r.beta, 0, nb * vh),
                        full(w(&p("ssm_dt.bias"))),
                        full(w(&p("ssm_a"))),
                        full(s0),
                        full(s1),
                        part(&r.z, 0, nb * cfg.inner),
                        full(w(&p("ssm_norm.weight"))),
                        part(&r.gated, 0, nb * cfg.inner),
                        full(&r.ctrl),
                    ];
                if nb == 3 && !prefill {
                    delta_b.push(full(s2));
                }
                b.simple("ssm_delta", if big { &k.big.as_ref().unwrap().delta[bidx(nb)] } else if prefill { &k.delta_chunk } else { &k.delta[nb - 1] }, delta_b, vh as u32);
                b.quant(r, k, &r.gated, nb * cfg.inner);
                b.matvec(g, r, k, &p("ssm_out.weight"), nb, (&r.mixed, 0))?;
            }
            LayerRes::Attn { kc, vc } => {
                b.matvec(g, r, k, &p("attn_q.weight"), nb, (&r.qfull, 0))?;
                b.matvec(g, r, k, &p("attn_k.weight"), nb, (&r.kin, 0))?;
                b.matvec(g, r, k, &p("attn_v.weight"), nb, (&r.vin, 0))?;
                attn_core(&mut b, r, k, cfg, nb, &p, kc, vc);
                b.quant(r, k, &r.attn_out, nb * cfg.n_head * cfg.head_dim);
                b.matvec(g, r, k, &p("attn_output.weight"), nb, (&r.mixed, 0))?;
            }
        }
        if ffn_block(&mut b, g, r, k, cfg, nb, il, defer)? {
            pending = Some(&r.down);
        }
        if let (Some(tb), false) = (&r.trace, prefill) {
            b.simple("copy", &k.copy, vec![part(&r.x, 0, nb * ne), part(tb, il * 3 * ne, nb * ne)], (nb * ne / 256) as u32);
        }
    }
    if pending.is_some() {
        b.norm_quant(r, k, ne, w("output_norm.weight"), nb, pending.take(), Some(&r.hfin));
    } else {
        b.simple("rmsnorm", &k.norm, vec![part(&r.x, 0, nb * ne), full(w("output_norm.weight")), part(&r.hfin, 0, nb * ne)], nb as u32);
        if !prefill {
            b.quant(r, k, &r.hfin, nb * ne);
        }
    }
    if !prefill {
        b.matvec(g, r, k, "output.weight", nb, (&r.logits, 0))?;
        b.simple("argmax", &k.argmax, vec![part(&r.logits, 0, nb * cfg.vocab), part(&r.am_main, 0, nb), part(&r.mg_main, 0, nb)], nb as u32);
    }
    Ok(b)
}

fn slot_buf(r: &Res, s: Slot) -> &Buf {
    match s {
        Slot::Mixed => &r.mixed,
        Slot::Down => &r.down,
        Slot::Gate => &r.gate,
        Slot::Up => &r.up,
        Slot::Qkv => &r.qkv,
        Slot::Z => &r.z,
        Slot::Beta => &r.beta,
        Slot::Alpha => &r.alpha,
        Slot::Gated => &r.gated,
        Slot::Qfull => &r.qfull,
        Slot::Kin => &r.kin,
        Slot::Vin => &r.vin,
        Slot::AttnOut => &r.attn_out,
        Slot::Logits => &r.logits,
    }
}

/// The same decode pass emitted from the kernel operations that `lower::lower` derived from the
/// model IR (no matrix-core or chunk variants: those stay hand-written).
fn emit_main_ir<'a>(g: &GgufFile, cfg: &Config, r: &'a Res, k: &'a Kernels, nb: usize, ops: &[KOp]) -> Result<Builder<'a>, Error> {
    let (ne, cd) = (cfg.n_embd, cfg.conv_dim());
    let (kh, vh, hs) = (cfg.k_heads, cfg.v_heads, cfg.state);
    let (nh, nkv) = (cfg.n_head, cfg.n_kv);
    let mut b = Builder::new();
    // Wide batches go through the matrix-core GEMMs (f16 activations).
    b.gemm = BIG_SIZES.contains(&nb);
    let w = |n: &str| -> &'a Buf { &r.weights[n] };
    for op in ops {
        match op {
            KOp::Embed => b.simple("embed", &k.embed, vec![full(w("token_embd.weight")), part(&r.tok_main, 0, nb), part(&r.x, 0, nb * ne)], (nb * ne / 256) as u32),
            KOp::NormQuant { weight, add, keep_hidden } => b.norm_quant(r, k, ne, w(weight), nb, add.map(|s| slot_buf(r, s)), if *keep_hidden { Some(&r.hfin) } else { None }),
            KOp::Quant { src } => {
                let per_col = match src {
                    Slot::Gated => cfg.inner,
                    Slot::AttnOut => nh * cfg.head_dim,
                    other => return Err(format!("no quantization of slot {other:?}").into()),
                };
                b.quant(r, k, slot_buf(r, *src), nb * per_col);
            }
            KOp::Matvec { weight, out } => b.matvec(g, r, k, weight, nb, (slot_buf(r, *out), 0))?,
            KOp::SsmConv { layer, weight } => {
                let LayerRes::Ssm { c0, c1, c2, .. } = &r.layers[*layer] else { return Err(format!("layer {layer} is not a gated delta layer").into()) };
                let mut conv_b = vec![part(&r.qkv, 0, nb * cd), full(c0), full(c1), full(w(weight)), part(&r.conv_out, 0, nb * cd), full(&r.ctrl)];
                if nb == 3 {
                    conv_b.push(full(c2));
                }
                b.simple("ssm_conv", &k.conv[nb - 1], conv_b, (cd / 256) as u32);
            }
            KOp::SsmQkNorm => b.simple("ssm_qknorm", &k.qknorm, vec![part(&r.conv_out, 0, nb * cd), part(&r.qk, 0, nb * 2 * kh * hs)], (nb * 2 * kh) as u32),
            KOp::SsmDelta { layer, dt_bias, a, norm } => {
                let LayerRes::Ssm { s0, s1, s2, .. } = &r.layers[*layer] else { return Err(format!("layer {layer} is not a gated delta layer").into()) };
                let mut delta_b = vec![
                    part(&r.qk, 0, nb * 2 * kh * hs),
                    part(&r.conv_out, 0, nb * cd),
                    part(&r.alpha, 0, nb * vh),
                    part(&r.beta, 0, nb * vh),
                    full(w(dt_bias)),
                    full(w(a)),
                    full(s0),
                    full(s1),
                    part(&r.z, 0, nb * cfg.inner),
                    full(w(norm)),
                    part(&r.gated, 0, nb * cfg.inner),
                    full(&r.ctrl),
                ];
                if nb == 3 {
                    delta_b.push(full(s2));
                }
                b.simple("ssm_delta", &k.delta[nb - 1], delta_b, vh as u32);
            }
            KOp::AttnPrep { layer, q_norm, k_norm } => {
                let LayerRes::Attn { kc, vc } = &r.layers[*layer] else { return Err(format!("layer {layer} is not an attention layer").into()) };
                b.simple(
                    "attn_prep",
                    &k.prep,
                    vec![part(&r.qfull, 0, nb * nh * cfg.head_dim * 2), part(&r.kin, 0, nb * nkv * cfg.head_dim), part(&r.vin, 0, nb * nkv * cfg.head_dim), full(w(q_norm)), full(w(k_norm)), part(&r.q_out, 0, nb * nh * cfg.head_dim), full(kc), full(vc), full(&r.ctrl)],
                    (nb * (nh + 2 * nkv)) as u32,
                );
            }
            KOp::AttnDecode { layer } => {
                let LayerRes::Attn { kc, vc } = &r.layers[*layer] else { return Err(format!("layer {layer} is not an attention layer").into()) };
                b.simple(
                    "attn_decode",
                    &k.attn,
                    vec![part(&r.q_out, 0, nb * nh * cfg.head_dim), full(kc), full(vc), part(&r.qfull, 0, nb * nh * cfg.head_dim * 2), part(&r.attn_out, 0, nb * nh * cfg.head_dim), full(&r.ctrl)],
                    (nb * nh) as u32,
                );
            }
            KOp::SsmConvBatch { layer, weight } => {
                let (Some(bk), Some(LayerPool::Ssm { conv, .. })) = (&k.batch, r.pools.get(*layer)) else { return Err(format!("layer {layer}: no batch pool for the convolution").into()) };
                b.simple("ssm_conv", &bk.conv[bsidx(nb)], vec![part(&r.qkv, 0, nb * cd), full(conv), full(w(weight)), part(&r.conv_out, 0, nb * cd), full(&r.ctrl)], (nb * cd / 256) as u32);
            }
            KOp::SsmDeltaBatch { layer, dt_bias, a, norm } => {
                let (Some(bk), Some(LayerPool::Ssm { rec, .. })) = (&k.batch, r.pools.get(*layer)) else { return Err(format!("layer {layer}: no batch pool for the delta state").into()) };
                b.simple(
                    "ssm_delta",
                    &bk.delta,
                    vec![
                        part(&r.qk, 0, nb * 2 * kh * hs),
                        part(&r.conv_out, 0, nb * cd),
                        part(&r.alpha, 0, nb * vh),
                        part(&r.beta, 0, nb * vh),
                        full(w(dt_bias)),
                        full(w(a)),
                        full(rec),
                        part(&r.z, 0, nb * cfg.inner),
                        full(w(norm)),
                        part(&r.gated, 0, nb * cfg.inner),
                        full(&r.ctrl),
                    ],
                    (nb * vh) as u32,
                );
            }
            KOp::AttnPrepBatch { layer, q_norm, k_norm } => {
                let (Some(bk), Some(LayerPool::Attn { k: kp, v: vp })) = (&k.batch, r.pools.get(*layer)) else { return Err(format!("layer {layer}: no batch slab").into()) };
                b.simple(
                    "attn_prep",
                    &bk.prep,
                    vec![part(&r.qfull, 0, nb * nh * cfg.head_dim * 2), part(&r.kin, 0, nb * nkv * cfg.head_dim), part(&r.vin, 0, nb * nkv * cfg.head_dim), full(w(q_norm)), full(w(k_norm)), part(&r.q_out, 0, nb * nh * cfg.head_dim), full(kp), full(vp), full(&r.ctrl)],
                    (nb * (nh + 2 * nkv)) as u32,
                );
            }
            KOp::AttnDecodeBatch { layer } => {
                let (Some(bk), Some(LayerPool::Attn { k: kp, v: vp })) = (&k.batch, r.pools.get(*layer)) else { return Err(format!("layer {layer}: no batch slab").into()) };
                b.simple(
                    "attn_decode",
                    &bk.decode,
                    vec![part(&r.q_out, 0, nb * nh * cfg.head_dim), full(kp), full(vp), part(&r.qfull, 0, nb * nh * cfg.head_dim * 2), part(&r.attn_out, 0, nb * nh * cfg.head_dim), full(&r.ctrl)],
                    (nb * nh) as u32,
                );
            }
            KOp::AttnPrepPaged { layer, q_norm, k_norm } => {
                let (Some(bk), Some(LayerPool::Attn { k: kp, v: vp })) = (&k.batch, r.pools.get(*layer)) else { return Err(format!("layer {layer}: no paged KV pool").into()) };
                b.simple(
                    "attn_prep",
                    &bk.prep_paged,
                    vec![part(&r.qfull, 0, nb * nh * cfg.head_dim * 2), part(&r.kin, 0, nb * nkv * cfg.head_dim), part(&r.vin, 0, nb * nkv * cfg.head_dim), full(w(q_norm)), full(w(k_norm)), part(&r.q_out, 0, nb * nh * cfg.head_dim), full(kp), full(vp), full(&r.ctrl)],
                    (nb * (nh + 2 * nkv)) as u32,
                );
            }
            KOp::AttnDecodePaged { layer } => {
                let (Some(bk), Some(LayerPool::Attn { k: kp, v: vp })) = (&k.batch, r.pools.get(*layer)) else { return Err(format!("layer {layer}: no paged KV pool").into()) };
                b.simple(
                    "attn_decode",
                    &bk.decode_paged,
                    vec![part(&r.q_out, 0, nb * nh * cfg.head_dim), full(kp), full(vp), part(&r.qfull, 0, nb * nh * cfg.head_dim * 2), part(&r.attn_out, 0, nb * nh * cfg.head_dim), full(&r.ctrl), full(&r.block_table)],
                    (nb * nh) as u32,
                );
            }
            KOp::SwigluQuant => b.swiglu_quant(r, k, nb, cfg.n_ff),
            KOp::Argmax => b.simple("argmax", &k.argmax, vec![part(&r.logits, 0, nb * cfg.vocab), part(&r.am_main, 0, nb), part(&r.mg_main, 0, nb)], nb as u32),
        }
    }
    Ok(b)
}

/// The two command streams must be identical: same kernels, buffers, ranges, constants, groups.
fn steps_equal(a: &Builder<'_>, b: &Builder<'_>) -> Result<(), String> {
    if a.steps.len() != b.steps.len() {
        return Err(format!("{} steps against {}", a.steps.len(), b.steps.len()));
    }
    for (i, (x, y)) in a.steps.iter().zip(&b.steps).enumerate() {
        let (lx, ly) = (&a.labels[i], &b.labels[i]);
        if !std::ptr::eq(x.kernel, y.kernel) {
            return Err(format!("step {i}: different kernels ({lx} against {ly})"));
        }
        if x.bindings.len() != y.bindings.len() || x.bindings.iter().zip(&y.bindings).any(|(p, q)| !std::ptr::eq(p.0, q.0) || p.1 != q.1 || p.2 != q.2) {
            return Err(format!("step {i} ({lx}): different bindings"));
        }
        if x.push != y.push || x.groups != y.groups {
            return Err(format!("step {i} ({lx}): different push constants or group count"));
        }
    }
    Ok(())
}

/// q/k norm + rope + KV write + attention + gate; reads `qfull`, `kin`, `vin`
/// and leaves the gated result in `attn_out`.
#[allow(clippy::too_many_arguments)]
fn attn_core<'a>(b: &mut Builder<'a>, r: &'a Res, k: &'a Kernels, cfg: &Config, nb: usize, p: &dyn Fn(&str) -> String, kc: &'a Buf, vc: &'a Buf) {
    let w = |n: &str| -> &'a Buf { &r.weights[n] };
    let (nh, nkv) = (cfg.n_head, cfg.n_kv);
    b.simple(
        "attn_prep",
        &k.prep,
        vec![part(&r.qfull, 0, nb * nh * cfg.head_dim * 2), part(&r.kin, 0, nb * nkv * cfg.head_dim), part(&r.vin, 0, nb * nkv * cfg.head_dim), full(w(&p("attn_q_norm.weight"))), full(w(&p("attn_k_norm.weight"))), part(&r.q_out, 0, nb * nh * cfg.head_dim), full(kc), full(vc), full(&r.ctrl)],
        (nb * (nh + 2 * nkv)) as u32,
    );
    b.simple(
        "attn_decode",
        &k.attn,
        vec![part(&r.q_out, 0, nb * nh * cfg.head_dim), full(kc), full(vc), part(&r.qfull, 0, nb * nh * cfg.head_dim * 2), part(&r.attn_out, 0, nb * nh * cfg.head_dim), full(&r.ctrl)],
        (nb * nh) as u32,
    );
}

/// post-attention residual, FFN and its residual for layer `il`. With `defer`
/// the closing residual add is left to the next norm kernel (`x += down`) and
/// the function returns true.
#[allow(clippy::too_many_arguments)]
fn ffn_block<'a>(b: &mut Builder<'a>, g: &GgufFile, r: &'a Res, k: &'a Kernels, cfg: &Config, nb: usize, il: usize, defer: bool) -> Result<bool, Error> {
    let (ne, nff) = (cfg.n_embd, cfg.n_ff);
    let p = |n: &str| format!("blk.{il}.{n}");
    b.norm_quant(r, k, ne, &r.weights[&p("post_attention_norm.weight")], nb, Some(&r.mixed), None);
    b.matvec(g, r, k, &p("ffn_gate.weight"), nb, (&r.gate, 0))?;
    b.matvec(g, r, k, &p("ffn_up.weight"), nb, (&r.up, 0))?;
    b.swiglu_quant(r, k, nb, nff);
    b.matvec(g, r, k, &p("ffn_down.weight"), nb, (&r.down, 0))?;
    if defer {
        return Ok(true);
    }
    b.simple("add", &k.add, vec![part(&r.x, 0, nb * ne), part(&r.down, 0, nb * ne)], (nb * ne / 256) as u32);
    Ok(false)
}
/// MTP KV fill for `MAX_COLS` consecutive prompt positions at once: column `c`
/// pairs the main hidden state of position `pos+c` (`hfin`, from a prefill
/// chunk) with the embedding of the token at `pos+c+1` (`tok_mtp`).
fn build_mtp_fill_chunk(ctx: &VulkanContext, g: &GgufFile, cfg: &Config, r: &Res, k: &Kernels, nb: usize, big: bool) -> Result<Program, Error> {
    let ne = cfg.n_embd;
    let il = cfg.n_trunk;
    let p = |n: &str| format!("blk.{il}.{n}");
    let w = |n: &str| -> &Buf { &r.weights[n] };
    let mut b = Builder::new();
    b.gemm = big;
    b.simple("embed", &k.embed, vec![full(w("token_embd.weight")), part(&r.tok_mtp, 0, nb), part(&r.e_in, 0, nb * ne)], (nb * ne / 256) as u32);
    b.simple("rmsnorm", &k.norm, vec![part(&r.e_in, 0, nb * ne), full(w(&p("nextn.enorm.weight"))), part(&r.xn, 0, nb * ne)], nb as u32);
    b.simple("cat", &k.cat_e, vec![part(&r.xn, 0, nb * ne), part(&r.cat, 0, nb * 2 * ne)], (nb * ne / 256) as u32);
    b.simple("rmsnorm", &k.norm, vec![part(&r.hfin, 0, nb * ne), full(w(&p("nextn.hnorm.weight"))), part(&r.xn, 0, nb * ne)], nb as u32);
    b.simple("cat", &k.cat_h, vec![part(&r.xn, 0, nb * ne), part(&r.cat, 0, nb * 2 * ne)], (nb * ne / 256) as u32);
    b.quant(r, k, &r.cat, nb * 2 * ne);
    b.matvec(g, r, k, &p("nextn.eh_proj.weight"), nb, (&r.x, 0))?;
    b.simple("rmsnorm", &k.norm, vec![part(&r.x, 0, nb * ne), full(w(&p("attn_norm.weight"))), part(&r.xn, 0, nb * ne)], nb as u32);
    b.quant(r, k, &r.xn, nb * ne);
    b.matvec(g, r, k, &p("attn_k.weight"), nb, (&r.kin, 0))?;
    b.matvec(g, r, k, &p("attn_v.weight"), nb, (&r.vin, 0))?;
    let (nh, nkv, hd) = (cfg.n_head, cfg.n_kv, cfg.head_dim);
    b.simple(
        "attn_prep",
        &k.prep,
        vec![part(&r.qfull, 0, nb * nh * hd * 2), part(&r.kin, 0, nb * nkv * hd), part(&r.vin, 0, nb * nkv * hd), full(w(&p("attn_q_norm.weight"))), full(w(&p("attn_k_norm.weight"))), part(&r.q_out, 0, nb * nh * hd), full(&r.mtp_kv.0), full(&r.mtp_kv.1), full(&r.ctrl)],
        (nb * (nh + 2 * nkv)) as u32,
    );
    Ok(ctx.build_program(&b.steps, false)?)
}
/// MTP head (layer `n_trunk`): `cat = [rmsnorm(embed), rmsnorm(h)]`, `eh_proj`,
/// one full-attention block with FFN, then (if `draft`) the shared output head.
/// `h_src` replaces the main hidden state by another buffer (chained draft); `save` copies the
/// block's output hidden state (before the shared head norm with LEV_CHAIN_SRC=x, after it otherwise) there.
fn build_mtp(ctx: &VulkanContext, g: &GgufFile, cfg: &Config, r: &Res, k: &Kernels, draft: bool, hcol: usize, h_src: Option<&Buf>, save: Option<&Buf>) -> Result<Program, Error> {
    let ne = cfg.n_embd;
    let il = cfg.n_trunk;
    let p = |n: &str| format!("blk.{il}.{n}");
    let w = |n: &str| -> &Buf { &r.weights[n] };
    let mut b = Builder::new();
    b.simple("embed", &k.embed, vec![full(w("token_embd.weight")), part(&r.tok_mtp, 0, 1), part(&r.e_in, 0, ne)], (ne / 256) as u32);
    b.simple("rmsnorm", &k.norm, vec![part(&r.e_in, 0, ne), full(w(&p("nextn.enorm.weight"))), part(&r.cat, 0, ne)], 1);
    b.simple("rmsnorm", &k.norm, vec![match h_src { Some(h) => part(h, 0, ne), None => part(&r.hfin, hcol * ne, ne) }, full(w(&p("nextn.hnorm.weight"))), part(&r.cat, ne, ne)], 1);
    b.quant(r, k, &r.cat, 2 * ne);
    b.matvec(g, r, k, &p("nextn.eh_proj.weight"), 1, (&r.x, 0))?;
    b.simple("rmsnorm", &k.norm, vec![part(&r.x, 0, ne), full(w(&p("attn_norm.weight"))), part(&r.xn, 0, ne)], 1);
    b.quant(r, k, &r.xn, ne);
    if draft {
        b.matvec(g, r, k, &p("attn_q.weight"), 1, (&r.qfull, 0))?;
    }
    b.matvec(g, r, k, &p("attn_k.weight"), 1, (&r.kin, 0))?;
    b.matvec(g, r, k, &p("attn_v.weight"), 1, (&r.vin, 0))?;
    if draft {
        attn_core(&mut b, r, k, cfg, 1, &p, &r.mtp_kv.0, &r.mtp_kv.1);
        b.quant(r, k, &r.attn_out, cfg.n_head * cfg.head_dim);
        b.matvec(g, r, k, &p("attn_output.weight"), 1, (&r.mixed, 0))?;
        ffn_block(&mut b, g, r, k, cfg, 1, il, false)?;
        b.simple("rmsnorm", &k.norm, vec![part(&r.x, 0, ne), full(w(&p("nextn.shared_head_norm.weight"))), part(&r.xn, 0, ne)], 1);
        if let Some(sv) = save {
            let from = if std::env::var("LEV_CHAIN_SRC").as_deref() == Ok("x") { &r.x } else { &r.xn };
            b.simple("copy", &k.copy, vec![part(from, 0, ne), part(sv, 0, ne)], (ne / 256) as u32);
        }
        b.quant(r, k, &r.xn, ne);
        b.matvec_rows(g, r, k, "output.weight", 1, (&r.mtp_logits, 0), mtp_vocab(cfg))?;
        b.simple("argmax", &k.argmax_mtp, vec![part(&r.mtp_logits, 0, mtp_vocab(cfg)), part(&r.am_mtp, 0, 1), part(&r.mg_mtp, 0, 1)], 1);
    } else {
        // Only this position's KV entry is needed: prep writes k and v (its q
        // output is unused scratch).
        attn_prep_only(&mut b, r, k, cfg, &p, &r.mtp_kv.0, &r.mtp_kv.1);
    }
    Ok(ctx.build_program(&b.steps, false)?)
}

fn attn_prep_only<'a>(b: &mut Builder<'a>, r: &'a Res, k: &'a Kernels, cfg: &Config, p: &dyn Fn(&str) -> String, kc: &'a Buf, vc: &'a Buf) {
    let (nh, nkv, hd) = (cfg.n_head, cfg.n_kv, cfg.head_dim);
    b.simple(
        "attn_prep",
        &k.prep,
        vec![part(&r.qfull, 0, nh * hd * 2), part(&r.kin, 0, nkv * hd), part(&r.vin, 0, nkv * hd), full(&r.weights[&p("attn_q_norm.weight")]), full(&r.weights[&p("attn_k_norm.weight")]), part(&r.q_out, 0, nh * hd), full(kc), full(vc), full(&r.ctrl)],
        (nh + 2 * nkv) as u32,
    );
}

/// Counters from a generation.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    pub steps: usize,
    pub accepted: usize,
    pub tokens: usize,
    pub seconds: f64,
    /// Time spent on the prompt (token by token) before the first generated token.
    pub prefill_seconds: f64,
    /// Summed GPU program time (device timestamps), microseconds.
    pub main_us: f64,
    pub mtp_us: f64,
    /// Plain decoding only: best-vs-second logit gap of each generated token.
    pub margins: Vec<f32>,
}

impl GpuModel {
    /// Loads all weights to the device and records the programs. `ctx_len`
    /// bounds the sequence length; `trace` additionally records the residual
    /// stream after every layer (single-column pass only).
    pub fn open(path: &str, ctx_len: usize, trace: bool) -> Result<GpuModel, Error> {
        Self::open_with(path, ctx_len, trace, false)
    }

    /// Like `open`, optionally recording a timestamp after every dispatch of
    /// the single-column program.
    pub fn open_with(path: &str, ctx_len: usize, trace: bool, profile: bool) -> Result<GpuModel, Error> {
        let gguf = GgufFile::open(path)?;
        let cfg = Config::from_gguf(&gguf)?;
        let mut file = File::open(path)?;
        let ctx = VulkanContext::open(&DeviceSelector::Auto)?;
        if !ctx.info().int_dot {
            return Err("this device lacks VK_KHR_shader_integer_dot_product (the float matvec path is not wired into the model)".into());
        }
        let dir = cache_dir();
        let mk = |src: &str, nb: usize, push: u32| -> Result<Kernel, Error> {
            let (spv, _) = compile_glsl_cached(src, &dir)?;
            Ok(ctx.create_kernel(&spv, nb, push)?)
        };
        let mk_sub = |src: &str, nb: usize, push: u32, wave: u32| -> Result<Kernel, Error> {
            let (spv, _) = compile_glsl_cached(src, &dir)?;
            Ok(ctx.create_kernel_sub(&spv, nb, push, Some(wave))?)
        };
        let (hd, nh, nkv) = (cfg.head_dim, cfg.n_head, cfg.n_kv);
        let (hs, kh, vh, cd) = (cfg.state, cfg.k_heads, cfg.v_heads, cfg.conv_dim());
        // Ragged batch decoding: number of sequence slots (LEV_BATCH_SLOTS, 0 = off).
        let batch_slots: usize = std::env::var("LEV_BATCH_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let kernels = Kernels {
            norm: mk(&ops::rmsnorm(cfg.n_embd, cfg.eps), 3, 0)?,
            add: mk(&ops::add_inplace(), 2, 0)?,
            swiglu: mk(&ops::swiglu(), 3, 0)?,
            copy: mk(&ops::copy(), 2, 0)?,
            qknorm: mk(&ops::ssm_qk_norm(kh, hs, cd), 2, 0)?,
            prep: mk(&ops::attn_prep(nh, nkv, hd, cfg.rope_dim, cfg.eps), 9, 0)?,
            attn: mk(&ops::attn_decode(nh, nkv, hd, ctx_len), 6, 0)?,
            quant: mk(&q8::emit_quantize(), 3, 0)?,
            norm_q: mk(&q8::emit_norm_quant(cfg.n_embd, cfg.eps, false, false), 4, 0)?,
            add_norm_q: mk(&q8::emit_norm_quant(cfg.n_embd, cfg.eps, true, false), 5, 0)?,
            add_norm_q_y: mk(&q8::emit_norm_quant(cfg.n_embd, cfg.eps, true, true), 6, 0)?,
            swiglu_q: mk(&q8::emit_swiglu_quant(), 4, 0)?,
            argmax: mk(&ops::argmax(cfg.vocab), 3, 0)?,
            argmax_mtp: mk(&ops::argmax(mtp_vocab(&cfg)), 3, 0)?,
            embed: mk(&ops::embed_q4k(cfg.n_embd), 3, 0)?,
            conv: [mk(&ops::ssm_conv(cd, cfg.conv_k, 1), 6, 0)?, mk(&ops::ssm_conv(cd, cfg.conv_k, 2), 6, 0)?, mk(&ops::ssm_conv(cd, cfg.conv_k, 3), 7, 0)?],
            delta: [mk(&ops::ssm_delta(kh, vh, hs, cfg.eps, cd, 1), 12, 0)?, mk(&ops::ssm_delta(kh, vh, hs, cfg.eps, cd, 2), 12, 0)?, mk(&ops::ssm_delta(kh, vh, hs, cfg.eps, cd, 3), 13, 0)?],
            conv_chunk: mk(&ops::ssm_conv_chunk(cd, cfg.conv_k, MAX_COLS), 6, 0)?,
            delta_chunk: mk(&ops::ssm_delta_chunk(kh, vh, hs, cfg.eps, cd, MAX_COLS), 12, 0)?,
            cat_e: mk(&ops::copy_cols(cfg.n_embd, 2 * cfg.n_embd, 0), 2, 0)?,
            cat_h: mk(&ops::copy_cols(cfg.n_embd, 2 * cfg.n_embd, cfg.n_embd), 2, 0)?,
            big: if ctx.info().coopmat {
                Some(BigKernels {
                    act16: mk(&lev_forge::coopmat::emit_act_f16(), 2, 0)?,
                    gemm_q4: [mk_sub(&lev_forge::coopmat::emit_q4k_gemm_blocked_tokens(BIG_SIZES[0]), 3, 8, GEMM_WAVE as u32)?, mk_sub(&lev_forge::coopmat::emit_q4k_gemm_blocked_tokens(BIG_SIZES[1]), 3, 8, GEMM_WAVE as u32)?],
                    gemm_q6: [mk_sub(&lev_forge::coopmat::emit_q6k_gemm_blocked_tokens(BIG_SIZES[0]), 3, 8, GEMM_WAVE as u32)?, mk_sub(&lev_forge::coopmat::emit_q6k_gemm_blocked_tokens(BIG_SIZES[1]), 3, 8, GEMM_WAVE as u32)?],
                    conv: [mk(&ops::ssm_conv_chunk(cd, cfg.conv_k, BIG_SIZES[0]), 6, 0)?, mk(&ops::ssm_conv_chunk(cd, cfg.conv_k, BIG_SIZES[1]), 6, 0)?],
                    delta: [mk(&ops::ssm_delta_chunk(kh, vh, hs, cfg.eps, cd, BIG_SIZES[0]), 12, 0)?, mk(&ops::ssm_delta_chunk(kh, vh, hs, cfg.eps, cd, BIG_SIZES[1]), 12, 0)?],
                })
            } else {
                None
            },
            q4: [mk(&q8::emit_q4k_q8(&q4_schedule(1))?, 4, 8)?, mk(&q8::emit_q4k_q8(&q4_schedule(2))?, 4, 8)?, mk(&q8::emit_q4k_q8(&q4_schedule(8))?, 4, 8)?, mk(&q8::emit_q4k_q8(&q4_schedule(3))?, 4, 8)?, mk(&q8::emit_q4k_q8(&q4_schedule(4))?, 4, 8)?, mk(&q8::emit_q4k_q8(&q4_schedule(16))?, 4, 8)?],
            q6: [mk(&q8::emit_q6k_q8(&q6_schedule(1))?, 4, 8)?, mk(&q8::emit_q6k_q8(&q6_schedule(2))?, 4, 8)?, mk(&q8::emit_q6k_q8(&q6_schedule(8))?, 4, 8)?, mk(&q8::emit_q6k_q8(&q6_schedule(3))?, 4, 8)?, mk(&q8::emit_q6k_q8(&q6_schedule(4))?, 4, 8)?, mk(&q8::emit_q6k_q8(&q6_schedule(16))?, 4, 8)?],
            batch: if batch_slots > 0 {
                let conv = |nb: usize| mk(&ops::ssm_conv_batch(cd, cfg.conv_k, nb), 5, 0);
                Some(BatchKernels {
                    conv: [conv(1)?, conv(2)?, conv(4)?, conv(8)?, conv(16)?],
                    delta: mk(&ops::ssm_delta_batch(kh, vh, hs, cfg.eps, cd), 11, 0)?,
                    prep: mk(&ops::attn_prep_batch(nh, nkv, hd, cfg.rope_dim, cfg.eps, ctx_len), 9, 0)?,
                    decode: mk(&ops::attn_decode_batch(nh, nkv, hd, ctx_len), 6, 0)?,
                    prep_paged: mk(&ops::attn_prep_paged(nh, nkv, hd, cfg.rope_dim, cfg.eps), 9, 0)?,
                    decode_paged: mk(&ops::attn_decode_paged(nh, nkv, hd, ctx_len.div_ceil(ops::PAGE_SIZE)), 7, 0)?,
                })
            } else {
                None
            },
        };

        // Weights: one buffer per tensor.
        let mut weights: HashMap<String, Buf> = HashMap::new();
        let mut names: Vec<String> = Vec::new();
        // With LEV_IR=<ir.json> the trunk layers and their tensors come from the model IR (checked
        // against the operator sequences the kernels implement and against the model file); the
        // MTP block is not part of the IR yet and keeps its fixed list.
        let (ir_plan, ir_ops) = match load_ir(&cfg) {
            Ok(Some((plan, ops))) => {
                if std::env::var_os("LEV_VERBOSE").is_some() {
                    let ssm = plan.kinds.iter().filter(|k| **k == crate::plan::LayerKind::Ssm).count();
                    eprintln!("model IR: {} layers ({ssm} gated delta, {} attention), {} tensors, {} kernel operations", plan.kinds.len(), plan.kinds.len() - ssm, plan.tensors.len(), ops.len());
                }
                (Some(plan), Some(ops))
            }
            Ok(None) => (None, None),
            Err(e) => return Err(e),
        };
        for il in 0..=cfg.n_trunk {
            if il < cfg.n_trunk && ir_plan.is_some() {
                continue;
            }
            let recurrent = il < cfg.n_trunk && cfg.is_recurrent(il);
            let list: &[&str] = if recurrent {
                &["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_dt.bias", "ssm_a", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight"]
            } else {
                &["attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight"]
            };
            for n in list.iter().chain(["post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"].iter()) {
                names.push(format!("blk.{il}.{n}"));
            }
            if il == cfg.n_trunk {
                for n in ["nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"] {
                    names.push(format!("blk.{il}.{n}"));
                }
            }
        }
        match &ir_plan {
            Some(plan) => {
                for t in &plan.tensors {
                    if !names.contains(t) {
                        names.push(t.clone());
                    }
                }
            }
            None => {
                names.push("token_embd.weight".into());
                names.push("output_norm.weight".into());
                names.push("output.weight".into());
            }
        }
        for name in &names {
            let t = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
            let n = t.byte_size().ok_or("tensor size")?;
            let buf = ctx.create_device_buffer(n + 16)?;
            ctx.zero_buffer(&buf)?;
            let chunk = 256u64 << 20;
            let mut off = 0;
            while off < n {
                let len = chunk.min(n - off);
                ctx.upload(&buf, off, &gguf.read_tensor_bytes(&mut file, t, off, len)?)?;
                off += len;
            }
            weights.insert(name.clone(), buf);
        }

        let fb = |n: usize| -> Result<Buf, Error> { Ok(ctx.create_buffer((n * 4) as u64)?) };
        // Zeroed buffers that only the GPU touches (state, KV caches, scratch).
        let zb = |n: usize| -> Result<Buf, Error> {
            let b = ctx.create_device_buffer((n * 4) as u64)?;
            ctx.zero_buffer(&b)?;
            Ok(b)
        };
        let m = BIG_MAX;
        let (ne, nff) = (cfg.n_embd, cfg.n_ff);
        let mut layers = Vec::new();
        for il in 0..cfg.n_trunk {
            if cfg.is_recurrent(il) {
                layers.push(LayerRes::Ssm { c0: zb((cfg.conv_k - 1) * cd)?, c1: zb((cfg.conv_k - 1) * cd)?, c2: zb((cfg.conv_k - 1) * cd)?, s0: zb(vh * hs * hs)?, s1: zb(vh * hs * hs)?, s2: zb(vh * hs * hs)? });
            } else {
                layers.push(LayerRes::Attn { kc: zb(ctx_len * nkv * hd)?, vc: zb(ctx_len * nkv * hd)? });
            }
        }
        let ctrl = ctx.create_buffer(65536)?;
        ctrl.zero();
        let mut pools = Vec::new();
        // KV cache of the batch: paged blocks by default (LEV_KV_BLOCKS blocks, default as many as
        // slab slots would need), or one slab per slot with LEV_KV_SLAB=1.
        let kv_slab = std::env::var_os("LEV_KV_SLAB").is_some();
        let max_blocks = ctx_len.div_ceil(ops::PAGE_SIZE);
        let kv_blocks: usize = std::env::var("LEV_KV_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(batch_slots * max_blocks + 1).max(2);
        if batch_slots > 0 {
            let n = batch_slots + 1; // the last slot absorbs the padding columns
            for il in 0..cfg.n_trunk {
                if cfg.is_recurrent(il) {
                    pools.push(LayerPool::Ssm { conv: zb(n * (cfg.conv_k - 1) * cd)?, rec: zb(n * vh * hs * hs)? });
                } else if kv_slab {
                    pools.push(LayerPool::Attn { k: zb(n * ctx_len * nkv * hd)?, v: zb(n * ctx_len * nkv * hd)? });
                } else {
                    let floats = kv_blocks * ops::PAGE_SIZE * nkv * hd;
                    pools.push(LayerPool::Attn { k: zb(floats)?, v: zb(floats)? });
                }
            }
        }
        let block_table = ctx.create_buffer((((batch_slots + 1) * max_blocks).max(64) * 4) as u64)?;
        block_table.zero();
        let res = Res {
            weights,
            layers,
            pools,
            block_table,
            mtp_kv: (zb(ctx_len * nkv * hd)?, zb(ctx_len * nkv * hd)?),
            x: fb(m * ne)?,
            xn: zb(m * ne)?,
            mixed: zb(m * ne)?,
            gate: zb(m * nff)?,
            up: zb(m * nff)?,
            act: zb(m * nff)?,
            down: zb(m * ne)?,
            qkv: zb(m * cd)?,
            z: zb(m * cfg.inner)?,
            beta: zb(m * vh)?,
            alpha: zb(m * vh)?,
            conv_out: zb(m * cd)?,
            qk: zb(m * 2 * kh * hs)?,
            gated: zb(m * cfg.inner)?,
            qfull: zb(m * nh * hd * 2)?,
            kin: zb(m * nkv * hd)?,
            vin: zb(m * nkv * hd)?,
            q_out: zb(m * nh * hd)?,
            attn_out: zb(m * nh * hd)?,
            hfin: fb(m * ne)?,
            hmtp: fb(ne)?,
            tok_main: fb(m)?,
            tok_mtp: fb(m)?,
            am_main: fb(BATCH_MAX)?,
            mg_main: fb(BATCH_MAX)?,
            am_mtp: fb(1)?,
            mg_mtp: fb(1)?,
            xq: zb(BATCH_MAX.max(MAX_COLS) * nff.max(2 * ne))?,
            xs: zb(BATCH_MAX.max(MAX_COLS) * nff.max(2 * ne) / 32 * 2)?,
            xh: zb(m * nff.max(2 * ne) / 2)?,
            logits: fb(BATCH_MAX * cfg.vocab)?,
            ctrl,
            e_in: zb(m * ne)?,
            cat: zb(m * 2 * ne)?,
            mtp_logits: fb(cfg.vocab)?,
            trace: if trace { Some(fb(cfg.n_trunk * 3 * ne)?) } else { None },
        };

        // Decode passes: from the lowered model IR when there is one (after checking that its
        // command stream equals the hand-written pass), else the hand-written pass.
        let mk_main = |nb: usize, profile: bool| -> Result<(Program, Vec<String>), Error> {
            match &ir_ops {
                Some(ops) => {
                    let hand = emit_main(&gguf, &cfg, &res, &kernels, nb, false, false)?;
                    let low = emit_main_ir(&gguf, &cfg, &res, &kernels, nb, ops)?;
                    steps_equal(&hand, &low).map_err(|e| format!("the lowered model IR differs from the hand-written pass for {nb} columns: {e}"))?;
                    finish_program(&ctx, low, profile)
                }
                None => build_main(&ctx, &gguf, &cfg, &res, &kernels, nb, profile, false, false),
            }
        };
        let (main1, labels1) = mk_main(1, profile)?;
        let (main2, labels2) = mk_main(2, profile)?;
        let (main3, _) = mk_main(3, false)?;
        let mut batch_programs = Vec::new();
        if batch_slots > 0 {
            let mut ragged = load_ragged_ir(&cfg, !kv_slab)?;
            // Debug aid: keep only the first LEV_BATCH_TRUNC kernel operations.
            if let Some(n) = std::env::var("LEV_BATCH_TRUNC").ok().and_then(|v| v.parse::<usize>().ok()) {
                ragged.truncate(n);
            }
            for nb in BATCH_SIZES {
                if BIG_SIZES.contains(&nb) && kernels.big.is_none() {
                    continue; // no matrix cores: only the int8 sizes
                }
                let b = emit_main_ir(&gguf, &cfg, &res, &kernels, nb, &ragged)?;
                batch_programs.push((nb, finish_program(&ctx, b, false)?.0));
            }
        }
        let (prefill, labels_prefill) = build_main(&ctx, &gguf, &cfg, &res, &kernels, MAX_COLS, profile, true, false)?;
        let mtp_fill_chunk = build_mtp_fill_chunk(&ctx, &gguf, &cfg, &res, &kernels, MAX_COLS, false)?;
        let mut bigs: Vec<(usize, Program, Program)> = Vec::new();
        let mut labels_big = Vec::new();
        if kernels.big.is_some() {
            for &n in &BIG_SIZES {
                let (p, l) = build_main(&ctx, &gguf, &cfg, &res, &kernels, n, profile, true, true)?;
                let f = build_mtp_fill_chunk(&ctx, &gguf, &cfg, &res, &kernels, n, true)?;
                if n == BIG_COLS {
                    labels_big = l;
                }
                bigs.push((n, p, f));
            }
        }
        let save = Some(&res.hmtp);
        let mtp_draft = [build_mtp(&ctx, &gguf, &cfg, &res, &kernels, true, 0, None, save)?, build_mtp(&ctx, &gguf, &cfg, &res, &kernels, true, 1, None, save)?, build_mtp(&ctx, &gguf, &cfg, &res, &kernels, true, 2, None, save)?];
        let mtp_fill = [build_mtp(&ctx, &gguf, &cfg, &res, &kernels, false, 0, None, None)?, build_mtp(&ctx, &gguf, &cfg, &res, &kernels, false, 1, None, None)?, build_mtp(&ctx, &gguf, &cfg, &res, &kernels, false, 2, None, None)?];
        let mtp_chain = build_mtp(&ctx, &gguf, &cfg, &res, &kernels, true, 0, Some(&res.hmtp), None)?;
        let kv = if batch_slots > 0 && !kv_slab { Some(crate::kv::BlockManager::new(kv_blocks, batch_slots, max_blocks)) } else { None };
        Ok(GpuModel { main1, main2, main3, batch_programs, batch_slots, kv, prefill, mtp_fill_chunk, bigs, mtp_draft, mtp_chain, mtp_fill, labels1, labels2, labels_prefill, labels_big, res, _kernels: kernels, gguf, file, cfg, ctx_len, pos: 0, sel: 0, last_gpu_us: 0.0, last_mtp_us: 0.0, ctx })
    }

    pub fn dispatch_count(&self) -> usize {
        self.main1.steps
    }

    pub fn device_name(&self) -> &str {
        self.ctx.info().name.as_str()
    }

    pub fn labels(&self) -> &[String] {
        &self.labels1
    }

    /// Forgets the sequence: zeroes the recurrent state and rewinds to position 0.
    pub fn reset(&mut self) {
        for l in &self.res.layers {
            if let LayerRes::Ssm { c0, c1, c2, s0, s1, s2 } = l {
                for b in [c0, c1, c2, s0, s1, s2] {
                    let _ = self.ctx.zero_buffer(b);
                }
            }
        }
        self.pos = 0;
        self.sel = 0;
    }

    pub fn labels_prefill(&self) -> &[String] {
        &self.labels_prefill
    }

    /// Per-step GPU time of one prefill chunk (needs open_with(.., profile = true)).
    pub fn profile_chunk(&mut self, tokens: &[usize]) -> Result<Vec<f64>, Error> {
        assert_eq!(tokens.len(), MAX_COLS);
        self.res.tok_main.write_u32(0, &tokens.iter().map(|&t| t as u32).collect::<Vec<_>>());
        self.set_ctrl(self.pos, MAX_COLS);
        let v = self.prefill.run_profile()?;
        self.pos += MAX_COLS;
        self.sel = 0;
        Ok(v)
    }

    /// Driver statistics for the model's kernels (set LEV_PIPELINE_INFO before opening).
    pub fn kernel_reports(&self) -> Result<Vec<(String, Vec<String>)>, Error> {
        let k = &self._kernels;
        let mut list: Vec<(&str, &Kernel)> = vec![("rmsnorm", &k.norm), ("quantize", &k.quant), ("attn_prep", &k.prep), ("attn_decode", &k.attn), ("ssm_conv", &k.conv[0]), ("ssm_delta 1 col", &k.delta[0]), ("ssm_delta 2 col", &k.delta[1]), ("ssm_delta chunk 8", &k.delta_chunk), ("mv q4k int8 1 col", &k.q4[0]), ("mv q4k int8 8 col", &k.q4[2]), ("mv q6k int8 1 col", &k.q6[0]), ("mv q6k int8 8 col", &k.q6[2]), ("argmax", &k.argmax)];
        if let Some(b) = &k.big {
            list.extend([("ssm_delta chunk 64", &b.delta[0]), ("ssm_conv chunk 64", &b.conv[0]), ("gemm q4k 64", &b.gemm_q4[0]), ("gemm q6k 64", &b.gemm_q6[0]), ("gemm q4k 128", &b.gemm_q4[1]), ("gemm q6k 128", &b.gemm_q6[1])]);
        }
        let mut out = Vec::new();
        for (name, kernel) in list {
            out.push((name.to_string(), self.ctx.kernel_report(kernel, name, None)?));
        }
        Ok(out)
    }

    /// Contents of the main activation buffers (all columns), for debugging.
    pub fn debug_buffers(&self) -> Result<Vec<(String, Vec<f32>)>, Error> {
        let r = &self.res;
        let list: Vec<(&str, &Buf)> = vec![("x", &r.x), ("xn", &r.xn), ("qkv", &r.qkv), ("z", &r.z), ("beta", &r.beta), ("alpha", &r.alpha), ("conv_out", &r.conv_out), ("qk", &r.qk), ("gated", &r.gated), ("mixed", &r.mixed), ("gate", &r.gate), ("up", &r.up), ("act", &r.act), ("down", &r.down), ("qfull", &r.qfull), ("kin", &r.kin), ("vin", &r.vin), ("attn_out", &r.attn_out)];
        let mut out = Vec::new();
        for (name, b) in list {
            let bytes = if b.is_mapped() { let mut v = vec![0u8; b.size() as usize]; b.read_bytes(0, &mut v); v } else { self.ctx.download(b, 0, b.size())? };
            out.push((name.to_string(), bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()));
        }
        Ok(out)
    }

    pub fn labels_big(&self) -> &[String] {
        &self.labels_big
    }

    /// Per-step GPU time of one matrix-core chunk (needs profile mode).
    pub fn profile_big(&mut self, tokens: &[usize]) -> Result<Vec<f64>, Error> {
        assert_eq!(tokens.len(), BIG_COLS);
        self.res.tok_main.write_u32(0, &tokens.iter().map(|&t| t as u32).collect::<Vec<_>>());
        self.set_ctrl(self.pos, BIG_COLS);
        let v = self.bigs.first().ok_or("no matrix-core path")?.1.run_profile()?;
        self.pos += BIG_COLS;
        self.sel = 0;
        Ok(v)
    }

    pub fn labels_pair(&self) -> &[String] {
        &self.labels2
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    // ---- ragged batch decoding ----

    pub fn batch_slots(&self) -> usize {
        self.batch_slots
    }

    pub fn context_len(&self) -> usize {
        self.ctx_len
    }

    fn set_ctrl_batch(&self, items: &[BatchItem], nb: usize, flat: &[u32]) {
        let half = self.cfg.rope_dim / 2;
        let mut words = vec![0u32; ops::BATCH_SLOT_MAP + ops::CTRL_COLS];
        for c in 0..nb {
            // Columns beyond the batch are padding: the scratch slot at position 0.
            let it = items.get(c).copied().unwrap_or(BatchItem { slot: self.batch_slots, token: 0, pos: 0 });
            for i in 0..half {
                let theta = it.pos as f64 * self.cfg.rope_base.powf(-2.0 * i as f64 / self.cfg.rope_dim as f64);
                let (s, co) = theta.sin_cos();
                words[2 + c * ops::CTRL_WORDS_PER_COL + i] = (co as f32).to_bits();
                words[2 + c * ops::CTRL_WORDS_PER_COL + half + i] = (s as f32).to_bits();
            }
            words[ops::BATCH_META + c] = it.slot as u32;
            words[ops::BATCH_META + ops::CTRL_COLS + c] = it.pos as u32;
            words[ops::BATCH_SLOT_MAP + c] = flat.get(c).copied().unwrap_or(0); // padding writes the scratch block
        }
        self.res.ctrl.write_u32(0, &words);
    }

    /// One decode step for up to 16 sequences, each contributing its next token (distinct slots).
    /// Returns the greedy token and its logit margin per item. Needs LEV_BATCH_SLOTS.
    pub fn batch_step(&mut self, items: &[BatchItem]) -> Result<Vec<(usize, f32)>, Error> {
        if self.batch_slots == 0 {
            return Err("batch decoding is off (set LEV_BATCH_SLOTS)".into());
        }
        if items.is_empty() || items.len() > BATCH_MAX {
            return Err(format!("a batch holds 1 to {BATCH_MAX} tokens").into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for it in items {
            if it.slot >= self.batch_slots || !seen.insert(it.slot) {
                return Err(format!("slot {} is out of range or used twice in one step", it.slot).into());
            }
            if it.pos >= self.ctx_len || it.token >= self.cfg.vocab {
                return Err("position or token out of range".into());
            }
        }
        let nb = self.batch_programs.iter().map(|(s, _)| *s).find(|&s| s >= items.len()).ok_or("the batch is wider than the available programs")?;
        let mut toks: Vec<u32> = items.iter().map(|i| i.token as u32).collect();
        toks.resize(nb, 0);
        self.res.tok_main.write_u32(0, &toks);
        // Paged KV: every token needs its block; the table and the flat positions go to the kernels.
        let mut flat = Vec::new();
        if let Some(kv) = self.kv.as_mut() {
            for it in items {
                flat.push(kv.ensure(it.slot, it.pos).map_err(|e| format!("KV cache: {e:?}"))?);
            }
            self.res.block_table.write_u32(0, &kv.table());
        }
        self.set_ctrl_batch(items, nb, &flat);
        self.last_gpu_us = self.batch_programs.iter().find(|(s, _)| *s == nb).unwrap().1.run()?;
        Ok((0..items.len()).map(|c| self.main_choice(c)).collect())
    }

    /// Largest absolute difference between the states of two batch slots, per kind of state
    /// (convolution, recurrent, KV positions 0..`positions`); for checks.
    pub fn debug_slot_diff(&self, a: usize, b: usize, positions: usize) -> Result<Vec<(String, f32)>, Error> {
        let c = &self.cfg;
        let f32s = |bytes: Vec<u8>| -> Vec<f32> { bytes.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect() };
        let diff = |x: &[f32], y: &[f32]| x.iter().zip(y).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        let (cb, rb) = (((c.conv_k - 1) * c.conv_dim() * 4) as u64, (c.v_heads * c.state * c.state * 4) as u64);
        let kvrow = (c.n_kv * c.head_dim * 4) as u64;
        let (mut conv, mut rec, mut kv) = (0f32, 0f32, 0f32);
        for (il, pool) in self.res.pools.iter().enumerate() {
            match pool {
                LayerPool::Ssm { conv: cp, rec: rp } => {
                    conv = conv.max(diff(&f32s(self.ctx.download(cp, a as u64 * cb, cb)?), &f32s(self.ctx.download(cp, b as u64 * cb, cb)?)));
                    rec = rec.max(diff(&f32s(self.ctx.download(rp, a as u64 * rb, rb)?), &f32s(self.ctx.download(rp, b as u64 * rb, rb)?)));
                }
                LayerPool::Attn { k, .. } => {
                    let _ = il;
                    for p in 0..positions {
                        let at = |slot: usize| -> u64 {
                            match &self.kv {
                                Some(m) => (m.blocks_of(slot)[p / ops::PAGE_SIZE] as u64 * ops::PAGE_SIZE as u64 + (p % ops::PAGE_SIZE) as u64) * kvrow,
                                None => (slot as u64 * self.ctx_len as u64 + p as u64) * kvrow,
                            }
                        };
                        kv = kv.max(diff(&f32s(self.ctx.download(k, at(a), kvrow)?), &f32s(self.ctx.download(k, at(b), kvrow)?)));
                    }
                }
            }
        }
        Ok(vec![("conv".into(), conv), ("recurrent".into(), rec), ("kv".into(), kv)])
    }

    /// Logits of column 0 of the last main pass (host copy; for checks).
    pub fn last_logits(&self) -> Vec<f32> {
        self.main_logits(0)
    }

    /// Ends the sequence in `slot`: its KV blocks go back to the pool.
    pub fn batch_release(&mut self, slot: usize) {
        if let Some(kv) = self.kv.as_mut() {
            kv.release(slot);
        }
    }

    /// (free blocks, total blocks) of the paged KV pool, if there is one.
    pub fn kv_blocks(&self) -> Option<(usize, usize)> {
        self.kv.as_ref().map(|k| (k.free_blocks(), k.total_blocks()))
    }

    /// Prefills `prompt` into the main engine (fresh state) and returns the first generated token.
    pub fn prefill_prompt(&mut self, prompt: &[usize]) -> Result<usize, Error> {
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        self.reset();
        let mut i = 0;
        loop {
            let c = self.chunk_size(prompt.len() - i, true);
            if c == 0 {
                break;
            }
            self.prefill_chunk(&prompt[i..i + c], None)?;
            i += c;
        }
        for &t in &prompt[i..] {
            self.step(t)?;
        }
        Ok(self.main_choice(0).0)
    }

    /// Copies the main engine's current sequence (state after its prompt) into batch slot `slot`.
    pub fn batch_import(&mut self, slot: usize) -> Result<(), Error> {
        if slot >= self.batch_slots {
            return Err("slot out of range".into());
        }
        let c = &self.cfg;
        let (cd, kvrow) = (c.conv_dim(), c.n_kv * c.head_dim);
        let (conv_bytes, rec_bytes) = (((c.conv_k - 1) * cd * 4) as u64, (c.v_heads * c.state * c.state * 4) as u64);
        let slab = (self.ctx_len * kvrow * 4) as u64;
        let used = (self.pos * kvrow * 4) as u64;
        let sel = self.sel as usize;
        // Paged KV: the sequence gets the blocks of its positions; its old blocks are returned first.
        let mut blocks: Vec<u32> = Vec::new();
        if let Some(kv) = self.kv.as_mut() {
            kv.release(slot);
            if self.pos > 0 {
                kv.ensure(slot, self.pos - 1).map_err(|e| format!("KV cache: {e:?}"))?;
            }
            blocks = kv.blocks_of(slot).to_vec();
        }
        let paged = self.kv.is_some();
        for (layer, pool) in self.res.layers.iter().zip(&self.res.pools) {
            match (layer, pool) {
                (LayerRes::Ssm { c0, c1, c2, s0, s1, s2 }, LayerPool::Ssm { conv, rec }) => {
                    let (cb, sb) = ([c0, c1, c2][sel], [s0, s1, s2][sel]);
                    self.ctx.copy_buffer(cb, 0, conv, slot as u64 * conv_bytes, conv_bytes)?;
                    self.ctx.copy_buffer(sb, 0, rec, slot as u64 * rec_bytes, rec_bytes)?;
                }
                (LayerRes::Attn { kc, vc }, LayerPool::Attn { k, v }) if paged => {
                    let block_bytes = (ops::PAGE_SIZE * kvrow * 4) as u64;
                    for (i, &blk) in blocks.iter().enumerate() {
                        let len = block_bytes.min(used - i as u64 * block_bytes);
                        self.ctx.copy_buffer(kc, i as u64 * block_bytes, k, blk as u64 * block_bytes, len)?;
                        self.ctx.copy_buffer(vc, i as u64 * block_bytes, v, blk as u64 * block_bytes, len)?;
                    }
                }
                (LayerRes::Attn { kc, vc }, LayerPool::Attn { k, v }) => {
                    if used > 0 {
                        self.ctx.copy_buffer(kc, 0, k, slot as u64 * slab, used)?;
                        self.ctx.copy_buffer(vc, 0, v, slot as u64 * slab, used)?;
                    }
                }
                _ => return Err("layer and pool kinds differ".into()),
            }
        }
        Ok(())
    }

    fn set_ctrl(&self, pos_base: usize, cols: usize) {
        let half = self.cfg.rope_dim / 2;
        let mut words = vec![0u32; 2 + cols * ops::CTRL_WORDS_PER_COL];
        words[0] = self.sel;
        words[1] = pos_base as u32;
        for c in 0..cols {
            for i in 0..half {
                let theta = (pos_base + c) as f64 * self.cfg.rope_base.powf(-2.0 * i as f64 / self.cfg.rope_dim as f64);
                let (s, co) = theta.sin_cos();
                words[2 + c * ops::CTRL_WORDS_PER_COL + i] = (co as f32).to_bits();
                words[2 + c * ops::CTRL_WORDS_PER_COL + half + i] = (s as f32).to_bits();
            }
        }
        self.res.ctrl.write_u32(0, &words);
    }

    /// Runs `tokens` (1 or 2) at the next positions without advancing the
    /// position; the greedy choice of each column is read back with
    /// `main_choice`. The caller commits the position and the state selector
    /// once it knows how many columns to keep.
    fn run_main(&mut self, tokens: &[usize]) -> Result<(), Error> {
        let cols = tokens.len();
        if cols == 0 || cols > MAX_COLS || self.pos + cols > self.ctx_len {
            return Err("bad column count or context length exceeded".into());
        }
        if tokens.iter().any(|&t| t >= self.cfg.vocab) {
            return Err("token id out of range".into());
        }
        self.res.tok_main.write_u32(0, &tokens.iter().map(|&t| t as u32).collect::<Vec<_>>());
        self.set_ctrl(self.pos, cols);
        self.last_gpu_us = match cols {
            1 => self.main1.run()?,
            2 => self.main2.run()?,
            3 => self.main3.run()?,
            _ => return Err("unsupported column count".into()),
        };
        Ok(())
    }

    /// Greedy token and logit margin of column `c` of the last main pass.
    fn main_choice(&self, c: usize) -> (usize, f32) {
        let mut tok = [0u8; 4];
        self.res.am_main.read_bytes(c * 4, &mut tok);
        (u32::from_le_bytes(tok) as usize, self.res.mg_main.read_f32(c, 1)[0])
    }

    /// Logits of column `c` of the last main pass (host copy; slow, for checks).
    fn main_logits(&self, c: usize) -> Vec<f32> {
        self.res.logits.read_f32(c * self.cfg.vocab, self.cfg.vocab)
    }

    /// Single-token forward at the next position; returns the logits.
    pub fn forward(&mut self, token: usize) -> Result<Vec<f32>, Error> {
        self.step(token)?;
        Ok(self.main_logits(0))
    }

    /// Single-token forward without reading the logits back.
    fn step(&mut self, token: usize) -> Result<(), Error> {
        self.run_main(&[token])?;
        self.pos += 1;
        self.sel = 0;
        Ok(())
    }

    /// Per-step GPU time of one token (needs `open_with(.., profile = true)`).
    pub fn profile_token(&mut self, token: usize) -> Result<Vec<f64>, Error> {
        self.res.tok_main.write_u32(0, &[token as u32]);
        self.set_ctrl(self.pos, 1);
        let v = self.main1.run_profile()?;
        self.pos += 1;
        self.sel = 0;
        Ok(v)
    }

    /// Per-step GPU time of a two-token pass (needs open_with(.., profile = true)).
    pub fn profile_pair(&mut self, a: usize, b: usize) -> Result<Vec<f64>, Error> {
        self.res.tok_main.write_u32(0, &[a as u32, b as u32]);
        self.set_ctrl(self.pos, 2);
        let v = self.main2.run_profile()?;
        self.pos += 2;
        self.sel = 1;
        Ok(v)
    }

    /// Residual stream after each layer for the last single-token pass.
    pub fn layer_outputs(&self) -> Option<Vec<Vec<f32>>> {
        self.layer_outputs_col(0)
    }

    /// Residual stream after each layer for column c of the last pass.
    pub fn layer_outputs_col(&self, c: usize) -> Option<Vec<Vec<f32>>> {
        let t = self.res.trace.as_ref()?;
        let n = self.cfg.n_embd;
        Some((0..self.cfg.n_trunk).map(|il| t.read_f32((il * 3 + c) * n, n)).collect())
    }

    /// Two tokens in one weight pass at the next two positions, then keeps
    /// both. Returns both logits. For tests against two single-token passes.
    pub fn forward_pair(&mut self, a: usize, b: usize) -> Result<Vec<Vec<f32>>, Error> {
        self.run_main(&[a, b])?;
        self.pos += 2;
        self.sel = 1;
        Ok(vec![self.main_logits(0), self.main_logits(1)])
    }

    /// Three tokens in one weight pass at the next three positions, then keeps the first `keep`
    /// (1..=3). Returns the logits of all three columns. For tests against single-token passes.
    pub fn forward_triple(&mut self, toks: [usize; 3], keep: usize) -> Result<Vec<Vec<f32>>, Error> {
        self.run_main(&toks)?;
        self.pos += keep;
        self.sel = (keep - 1) as u32;
        Ok(vec![self.main_logits(0), self.main_logits(1), self.main_logits(2)])
    }

    /// MTP head at position `pos`: embeds `token`, takes the main hidden state
    /// of column `hcol` from the last main pass. With `draft` returns the
    /// head's greedy token; otherwise only writes the head's KV entry.
    fn mtp(&mut self, hcol: usize, token: usize, pos: usize, draft: bool) -> Result<Option<usize>, Error> {
        self.res.tok_mtp.write_u32(0, &[token as u32]);
        self.set_ctrl(pos, 1);
        if draft {
            self.last_mtp_us = self.mtp_draft[hcol].run()?;
            let mut tok = [0u8; 4];
            self.res.am_mtp.read_bytes(0, &mut tok);
            Ok(Some(u32::from_le_bytes(tok) as usize))
        } else {
            self.last_mtp_us = self.mtp_fill[hcol].run()?;
            Ok(None)
        }
    }

    /// Second draft chained from the block's own output (saved by the last draft pass): the token
    /// after `d1`, with the head's KV entry at position `pos`.
    fn chain(&mut self, d1: usize, pos: usize) -> Result<usize, Error> {
        self.res.tok_mtp.write_u32(0, &[d1 as u32]);
        self.set_ctrl(pos, 1);
        self.mtp_chain.run()?;
        let mut tok = [0u8; 4];
        self.res.am_mtp.read_bytes(0, &mut tok);
        Ok(u32::from_le_bytes(tok) as usize)
    }

    /// Chunk size to use next with `remaining` tokens left: the matrix-core
    /// chunk when available, then the 8-column chunk, else 0 (single steps).
    /// `keep_last` leaves the final token for a single step (it needs logits).
    fn chunk_size(&self, remaining: usize, keep_last: bool) -> usize {
        let need = |c: usize| if keep_last { remaining > c } else { remaining >= c };
        let allowed = std::env::var_os("LEV_NO_BIG").is_none();
        let max_big: usize = std::env::var("LEV_MAX_BIG").ok().and_then(|v| v.parse().ok()).unwrap_or(BIG_MAX);
        if allowed {
            if let Some((n, _, _)) = self.bigs.iter().rev().find(|(n, _, _)| *n <= max_big && need(*n)) {
                return *n;
            }
        }
        if need(MAX_COLS) {
            MAX_COLS
        } else {
            0
        }
    }

    /// Feeds `tokens` with chunked passes where possible, without logits.
    pub fn feed(&mut self, tokens: &[usize]) -> Result<(), Error> {
        let mut i = 0;
        loop {
            let c = self.chunk_size(tokens.len() - i, false);
            if c == 0 {
                break;
            }
            self.prefill_chunk(&tokens[i..i + c], None)?;
            i += c;
        }
        for &t in &tokens[i..] {
            self.step(t)?;
        }
        Ok(())
    }

    /// Processes `tokens.len()` prompt tokens (`MAX_COLS` or `BIG_COLS`) in one
    /// weight pass (no logits). With `next_tokens` also fills the MTP head's KV
    /// for those positions: `next_tokens[c]` is the token that follows
    /// `tokens[c]`.
    fn prefill_chunk(&mut self, tokens: &[usize], next_tokens: Option<&[usize]>) -> Result<(), Error> {
        let n = tokens.len();
        assert!(n == MAX_COLS || BIG_SIZES.contains(&n));
        if self.pos + n > self.ctx_len {
            return Err("context length exceeded".into());
        }
        self.res.tok_main.write_u32(0, &tokens.iter().map(|&t| t as u32).collect::<Vec<_>>());
        self.set_ctrl(self.pos, n);
        let t_run = std::time::Instant::now();
        let (main, fill) = if n != MAX_COLS {
            let e = self.bigs.iter().find(|(s, _, _)| *s == n).expect("matrix-core prefill");
            (&e.1, &e.2)
        } else {
            (&self.prefill, &self.mtp_fill_chunk)
        };
        self.last_gpu_us = main.run()?;
        if std::env::var_os("LEV_TRACE_CHUNK").is_some() {
            eprintln!("chunk of {n} at pos {}: gpu {:.1} ms, wall {:.1} ms", self.pos, self.last_gpu_us / 1000.0, t_run.elapsed().as_secs_f64() * 1000.0);
        }
        if let (Some(nt), false) = (next_tokens, std::env::var_os("LEV_NO_MTP_CHUNK").is_some()) {
            self.res.tok_mtp.write_u32(0, &nt.iter().map(|&t| t as u32).collect::<Vec<_>>());
            self.last_mtp_us = fill.run()?;
        }
        self.pos += n;
        self.sel = 0;
        Ok(())
    }
    /// Greedy generation of `n_new` tokens after `prompt` (the model must be
    /// fresh). Plain decoding when `spec` is false; MTP speculative decoding
    /// otherwise. Returns exactly `n_new` tokens and statistics.
    pub fn generate(&mut self, prompt: &[usize], n_new: usize, spec: bool) -> Result<(Vec<usize>, SpecStats), Error> {
        if prompt.is_empty() || self.pos != 0 {
            return Err("generate needs a non-empty prompt and a fresh model".into());
        }
        let mut stats = SpecStats::default();
        let t0 = std::time::Instant::now();
        let mut out: Vec<usize> = Vec::new();
        // Whole chunks first (all but the last prompt token), then single steps.
        let mut i = 0;
        let chunked = std::env::var_os("LEV_NO_CHUNK").is_none();
        loop {
            let c = if chunked { self.chunk_size(prompt.len() - i, true) } else { 0 };
            if c == 0 {
                break;
            }
            let next = if spec { Some(&prompt[i + 1..i + 1 + c]) } else { None };
            self.prefill_chunk(&prompt[i..i + c], next)?;
            stats.main_us += self.last_gpu_us;
            stats.mtp_us += self.last_mtp_us;
            i += c;
        }
        for (j, &t) in prompt.iter().enumerate().skip(i) {
            self.step(t)?;
            stats.main_us += self.last_gpu_us;
            if spec && j + 1 < prompt.len() {
                self.mtp(0, prompt[j + 1], j, false)?;
                stats.mtp_us += self.last_mtp_us;
            }
        }
        stats.prefill_seconds = t0.elapsed().as_secs_f64();
        let (mut next, margin) = self.main_choice(0);
        out.push(next);
        stats.margins.push(margin);
        if !spec {
            while out.len() < n_new {
                self.step(next)?;
                stats.main_us += self.last_gpu_us;
                let (t, m) = self.main_choice(0);
                next = t;
                out.push(next);
                stats.margins.push(m);
                stats.steps += 1;
            }
            stats.tokens = out.len();
            stats.seconds = t0.elapsed().as_secs_f64();
            return Ok((out, stats));
        }
        // Draft for the token after `next`, from the last prompt position.
        let depth: usize = std::env::var("LEV_SPEC_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let last = self.pos - 1;
        let mut draft = self.mtp(0, next, last, true)?.unwrap();
        stats.mtp_us += self.last_mtp_us;
        let mut d2 = if depth >= 2 { Some(self.chain(draft, last + 1)?) } else { None };
        while out.len() < n_new {
            let base = self.pos;
            if let Some(second) = d2 {
                // Three columns: next, draft, second draft. Accepting k tokens keeps the state
                // after column k - 1 (`sel`), and column c's hidden state feeds the next draft.
                self.run_main(&[next, draft, second])?;
                let (a0, a1, a2) = (self.main_choice(0).0, self.main_choice(1).0, self.main_choice(2).0);
                stats.steps += 1;
                stats.main_us += self.last_gpu_us;
                if std::env::var_os("LEV_TRACE_SPEC").is_some() {
                    eprintln!("step base={base} next={next} draft={draft},{second} a={a0},{a1},{a2}");
                }
                let kept;
                if a0 != draft {
                    kept = 1;
                    out.push(a0);
                    next = a0;
                } else if a1 != second {
                    kept = 2;
                    stats.accepted += 1;
                    out.push(draft);
                    out.push(a1);
                    next = a1;
                } else {
                    kept = 3;
                    stats.accepted += 2;
                    out.push(draft);
                    out.push(second);
                    out.push(a2);
                    next = a2;
                }
                self.pos += kept;
                self.sel = (kept - 1) as u32;
                // KV entries of the accepted drafts, then the draft chain from the last kept column.
                if kept >= 2 {
                    self.mtp(0, draft, base, false)?;
                    stats.mtp_us += self.last_mtp_us;
                }
                if kept >= 3 {
                    self.mtp(1, second, base + 1, false)?;
                    stats.mtp_us += self.last_mtp_us;
                }
                let at = base + kept - 1;
                draft = self.mtp(kept - 1, next, at, true)?.unwrap();
                stats.mtp_us += self.last_mtp_us;
                d2 = Some(self.chain(draft, at + 1)?);
                continue;
            }
            self.run_main(&[next, draft])?;
            let (a0, a1) = (self.main_choice(0).0, self.main_choice(1).0);
            stats.steps += 1;
            stats.main_us += self.last_gpu_us;
            if std::env::var_os("LEV_TRACE_SPEC").is_some() {
                eprintln!("step base={base} next={next} draft={draft} a0={a0} a1={a1} sel={}", self.sel);
            }
            if a0 == draft {
                // Both columns are valid: the draft and the token after it.
                stats.accepted += 1;
                self.pos += 2;
                self.sel = 1;
                out.push(draft);
                out.push(a1);
                self.mtp(0, draft, base, false)?;
                stats.mtp_us += self.last_mtp_us;
                next = a1;
                draft = self.mtp(1, a1, base + 1, true)?.unwrap();
                stats.mtp_us += self.last_mtp_us;
            } else {
                self.pos += 1;
                self.sel = 0;
                out.push(a0);
                next = a0;
                draft = self.mtp(0, a0, base, true)?.unwrap();
                stats.mtp_us += self.last_mtp_us;
            }
        }        out.truncate(n_new);
        stats.tokens = out.len();
        stats.seconds = t0.elapsed().as_secs_f64();
        Ok((out, stats))
    }
}