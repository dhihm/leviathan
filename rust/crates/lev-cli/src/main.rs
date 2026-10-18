use std::collections::BTreeMap;
use std::fs::File;
use std::process::ExitCode;

use lev_cpu::{
    dequant_block_f64, matvec_q4k_f32, matvec_q4k_f64, max_rel_err, SplitMix64, Q4K_BLOCK_BYTES, QK_K,
};
use lev_gguf::{GgmlType, GgufFile, Value};
use lev_vulkan::{compile_glsl_cached, DeviceSelector, VulkanContext};

/// Component gate, the repository's existing contract.
const GATE: f64 = 1e-4;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("inspect") if args.len() == 3 => inspect(&args[2]),
        Some("check-q4k") if (4..=5).contains(&args.len()) => {
            let rows = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
            match check_q4k(&args[2], &args[3], rows) {
                Ok(true) => ExitCode::SUCCESS,
                Ok(false) => ExitCode::FAILURE,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("vk-devices") if args.len() == 2 => match VulkanContext::list_devices() {
            Ok(devices) => {
                for d in devices {
                    println!(
                        "[{}] {} ({}) subgroup={} max_invocations={} timestamp_period={}ns valid_bits={} int_dot={} accelerated={}",
                        d.index,
                        d.name,
                        d.device_type,
                        d.caps.subgroup_size,
                        d.caps.max_workgroup_invocations,
                        d.timestamp_period_ns,
                        d.timestamp_valid_bits,
                        d.int_dot,
                        d.int_dot_accelerated
                    );
                    println!("    extensions: {}", d.extensions.join(", "));
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("vk-q4k") if (4..=10).contains(&args.len()) => {
            let rows = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4096);
            match vk_q4k(&args[2], &args[3], rows, args.get(5).map(String::as_str)) {
                Ok(true) => ExitCode::SUCCESS,
                Ok(false) => ExitCode::FAILURE,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("emit-q4k") if (4..=10).contains(&args.len()) => {
            let (Ok(t), Ok(r)) = (args[2].parse::<u32>(), args[3].parse::<u32>()) else {
                eprintln!("error: t and r must be integers");
                return ExitCode::from(2);
            };
            let xv = args.get(4).and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
            let nb = args.get(9).and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
            let u = args.get(8).and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
            let sg = args.get(7).and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
            let rr = args.get(6).and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
            let m = args.get(5).and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
            match lev_forge::glsl::emit_q4k_matvec(&lev_forge::schedule::Q4kSchedule { t, r, xv, m, rr, sg, u, nb }) {
                Ok(src) => {
                    print!("{src}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("vk-q6k") if args.len() == 4 || args.len() == 5 => match args[3].parse::<usize>().map(|r| vk_q6k(&args[2], args.get(4).map_or("blk.0.ffn_down.weight", |s| s.as_str()), r)) {
            Ok(Ok(pass)) => if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE },
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
            Err(_) => ExitCode::from(2),
        },
        Some("ref-forward") if args.len() >= 4 => match ref_forward(&args[2], &args[3]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-check") if args.len() == 4 => match gpu_check(&args[2], &args[3]) {
            Ok(pass) => if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE },
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-bench") if args.len() == 5 => match args[4].parse::<usize>().map(|n| gpu_bench(&args[2], &args[3], n)) {
            Ok(Ok(())) => ExitCode::SUCCESS,
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
            Err(_) => ExitCode::from(2),
        },
        Some("gpu-profile") if args.len() == 3 => match gpu_profile(&args[2]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-spec") if args.len() == 5 => match args[4].parse::<usize>().map(|n| gpu_spec(&args[2], &args[3], n)) {
            Ok(Ok(pass)) => if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE },
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
            Err(_) => ExitCode::from(2),
        },
        Some("gpu-pair") if args.len() == 4 => match gpu_pair(&args[2], &args[3]) {
            Ok(pass) => if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE },
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("vk-mem") => match VulkanContext::open(&DeviceSelector::Auto) {
            Ok(ctx) => {
                print!("{}", ctx.memory_report());
                println!("cooperative matrix configurations (component type ids: 0=f16 1=f32 2=f64 3=i8 4=i16 5=i32 6=i64 7=u8 8=u16 9=u32; scope 3=subgroup):");
                for l in ctx.cooperative_matrix_report() {
                    println!("  {l}");
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("vk-q8") if (4..=6).contains(&args.len()) => match args[3].parse::<usize>().map(|r| vk_q8(&args[2], args.get(4).map_or("blk.0.ffn_gate.weight", |s| s.as_str()), r, args.get(5).and_then(|v| v.parse().ok()).unwrap_or(1))) {
            Ok(Ok(pass)) => if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE },
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
            Err(_) => ExitCode::from(2),
        },
        Some("gpu-top") if args.len() == 4 => match gpu_top(&args[2], &args[3]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("vk-coopmat") => match vk_coopmat() {
            Ok(pass) => if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE },
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("vk-gemm") if args.len() == 4 || args.len() == 5 => match args[3].parse::<usize>().map(|r| vk_gemm(&args[2], r, args.get(4).map_or("blk.0.ffn_gate.weight", |s| s.as_str()))) {
            Ok(Ok(pass)) => if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE },
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
            Err(_) => ExitCode::from(2),
        },
        Some("gpu-kernels") if args.len() == 3 => match lev_model::gpu::GpuModel::open(&args[2], 512, false).and_then(|g| g.kernel_reports()) {
            Ok(rep) => {
                for (name, lines) in rep {
                    let pick = |key: &str| lines.iter().find(|l| l.contains(key)).map(|l| l.split('=').next_back().unwrap_or("").trim().to_string()).unwrap_or_default();
                    println!("{:<22} vgpr {:>4}  sgpr {:>4}  lds {:>6} B  scratch {:>6} B", name, pick("numUsedVgprs"), pick("numUsedSgprs"), pick("ldsUsageSizeInBytes"), pick("scratchMemUsageInBytes"));
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("vk-delta") => match vk_delta() {
            Ok(pass) => if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE },
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-soak") if args.len() == 5 => match args[3].parse::<u64>().map(|secs| gpu_soak(&args[2], secs, &args[4])) {
            Ok(Ok(())) => ExitCode::SUCCESS,
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
            Err(_) => ExitCode::from(2),
        },
        Some("gpu-batch") if args.len() == 5 => match gpu_batch(&args[2], &args[3], &args[4]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("ir-run") if args.len() == 3 => match ir_run(&args[2]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("ir-gguf") if args.len() == 6 => match ir_gguf(&args[2], &args[3], &args[4], &args[5]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("ir-dump") if args.len() == 4 => match ir_dump(&args[2], &args[3]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("ir-check") if args.len() == 4 => match ir_check(&args[2], &args[3]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-triple") if args.len() == 4 => match gpu_triple(&args[2], &args[3]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-dbg") if args.len() == 4 => match gpu_dbg(&args[2], &args[3]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("gpu-trace") if args.len() == 5 => match gpu_trace(&args[2], &args[3], &args[4]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("trace-diff") if args.len() == 4 => {
            let rd = |p: &str| -> Vec<f32> { std::fs::read(p).unwrap_or_default().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
            let (a, b) = (rd(&args[2]), rd(&args[3]));
            for (l, (x, y)) in a.chunks(4096).zip(b.chunks(4096)).enumerate() {
                let r: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                println!("layer {l:>2}: max_rel_err {:.2e}", max_rel_err(y, &r).unwrap_or(f64::INFINITY));
            }
            ExitCode::SUCCESS
        }
        Some("vk-bw") if args.len() == 4 => match args[3].parse::<usize>().map(|r| vk_bw(&args[2], r)) {
            Ok(Ok(())) => ExitCode::SUCCESS,
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
            Err(_) => ExitCode::from(2),
        },
        _ => {
            eprintln!("       lev vk-bw <model.gguf> <rows>   (streaming-read probe on the first rows of blk.0.ffn_gate.weight)");
            eprintln!("       lev vk-q6k <model.gguf> <rows> [tensor, default blk.0.ffn_down.weight]");
            eprintln!("       lev ref-forward <model.gguf> <token ids, comma separated>   (CPU reference, prints top logits of the last token)");
            eprintln!("       lev gpu-check <model.gguf> <token ids>   (GPU vs CPU reference, per layer and logits)");
            eprintln!("       lev gpu-bench <model.gguf> <prompt token ids> <n generated>   (greedy decode speed)");
            eprintln!("       lev gpu-spec <model.gguf> <prompt ids> <n>   (plain vs MTP speculative greedy decoding)");
            eprintln!("       lev gpu-pair <model.gguf> <token ids>   (two-column pass vs two single passes)");
            eprintln!("       lev vk-q8 <model.gguf> <rows> [tensor] [nb 1|2]   (int8-activation Q4_K matvec sweep)");
            eprintln!("       lev gpu-soak <model.gguf> <seconds> <ids file>   (repeat prefill + decode, one CSV line per iteration)");
            eprintln!("usage: lev inspect <model.gguf>");
            eprintln!("       lev emit-q4k <t> <r> [xv 1|4] [m 0|1] [rr 1|2|4]");
            eprintln!("       lev vk-devices");
            eprintln!("       lev vk-q4k <model.gguf> <tensor> [rows] [device index or name]");
            eprintln!("       lev check-q4k <model.gguf> <tensor> [rows]");
            ExitCode::from(2)
        }
    }
}

fn show(v: &Value) -> String {
    let s = match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Str(x) => x.replace('\n', " "),
        Value::Array { elem_type, len, values: Some(vs) } => {
            let items: Vec<String> = vs.iter().map(show).collect();
            format!("[{}] (type{elem_type} x {len})", items.join(","))
        }
        Value::Array { elem_type, len, values: None } => format!("<array {len} x type{elem_type}>"),
    };
    if s.len() > 120 {
        format!("{}...", &s[..s.char_indices().nth(120).map_or(s.len(), |(i, _)| i)])
    } else {
        s
    }
}

fn inspect(path: &str) -> ExitCode {
    let g = match GgufFile::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("GGUF v{} tensors={} kv={}", g.version, g.tensors.len(), g.metadata.len());
    println!("----- metadata");
    for (k, v) in &g.metadata {
        println!("{k} = {}", show(v));
    }

    // Per-layer tensors are grouped by their name after `blk.<N>.`.
    let mut groups: BTreeMap<String, (String, Vec<u32>)> = BTreeMap::new();
    for t in &g.tensors {
        let dims = t.dims.iter().map(u64::to_string).collect::<Vec<_>>().join("x");
        let example = format!("{} [{dims}]", t.ty.name());
        let split = t.name.strip_prefix("blk.").and_then(|rest| {
            let (n, tail) = rest.split_once('.')?;
            Some((n.parse::<u32>().ok()?, tail.to_string()))
        });
        match split {
            Some((layer, tail)) => groups.entry(tail).or_insert((example, Vec::new())).1.push(layer),
            None => {
                groups.insert(format!("(global) {}", t.name), (example, Vec::new()));
            }
        }
    }
    println!("----- tensors");
    for (name, (example, layers)) in &groups {
        if layers.is_empty() {
            println!("{name:<34} {example}");
        } else {
            let list = if layers.len() <= 12 {
                layers.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
            } else {
                format!("{}..{}", layers[0], layers[layers.len() - 1])
            };
            println!("{name:<34} {example:<22} n={} layers={list}", layers.len());
        }
    }

    println!("----- extents");
    println!("data_offset = {}", g.data_offset);
    match g.total_tensor_bytes() {
        Some(total) => {
            println!("tensor bytes = {total}");
            println!("file length  = {}", g.file_len);
            println!("trailing bytes after last tensor = {}", g.file_len.saturating_sub(g.data_offset + total));
        }
        None => println!("tensor bytes = <overflow>"),
    }
    ExitCode::SUCCESS
}

/// Runs the Q4_K component gate on the first `max_rows` rows of a real tensor
/// and prints dequantized-weight statistics as a sanity check of the layout.
fn check_q4k(path: &str, name: &str, max_rows: usize) -> Result<bool, Box<dyn std::error::Error>> {
    let g = GgufFile::open(path)?;
    let t = g.tensor(name).ok_or("tensor not found")?;
    if t.ty != GgmlType::Q4K || t.dims.len() != 2 {
        return Err(format!("{name}: expected a 2-D Q4_K tensor, got {} {:?}", t.ty.name(), t.dims).into());
    }
    let (cols, total_rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let rows = max_rows.min(total_rows);
    let row_bytes = cols / QK_K * Q4K_BLOCK_BYTES;
    let mut file = File::open(path)?;
    let weights = g.read_tensor_bytes(&mut file, t, 0, (rows * row_bytes) as u64)?;

    let mut rng = SplitMix64(0x4C45_5649);
    let x: Vec<f32> = (0..cols).map(|_| rng.next_signed_unit()).collect();
    let got = matvec_q4k_f32(&weights, rows, cols, &x)?;
    let reference = matvec_q4k_f64(&weights, rows, cols, &x)?;

    let (mut n, mut sum, mut sumsq) = (0u64, 0f64, 0f64);
    let (mut lo, mut hi, mut non_finite) = (f64::INFINITY, f64::NEG_INFINITY, 0u64);
    let mut tmp = [0f64; QK_K];
    for block in weights.chunks_exact(Q4K_BLOCK_BYTES) {
        dequant_block_f64(block, &mut tmp);
        for &v in &tmp {
            if v.is_finite() {
                n += 1;
                sum += v;
                sumsq += v * v;
                lo = lo.min(v);
                hi = hi.max(v);
            } else {
                non_finite += 1;
            }
        }
    }
    let mean = sum / n.max(1) as f64;
    let std = (sumsq / n.max(1) as f64 - mean * mean).max(0.0).sqrt();
    println!("tensor {name}: Q4_K [{cols} x {total_rows}], checking first {rows} rows");
    println!("dequantized weights: n={n} mean={mean:.6} std={std:.6} min={lo:.6} max={hi:.6} non_finite={non_finite}");

    let err = max_rel_err(&got, &reference);
    match err {
        Some(e) => println!("f32 vs f64 oracle: max relative error = {e:.3e} (gate < {GATE:e})"),
        None => println!("f32 vs f64 oracle: REJECTED (non-finite or length mismatch)"),
    }
    let pass = non_finite == 0 && err.is_some_and(|e| e < GATE);
    println!("{}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

/// Forge loop for a Q6_K tensor: derive, emit, compile (cached), run, gate
/// against the FP64 oracle, time.
fn vk_q6k(path: &str, name: &str, max_rows: usize) -> Result<bool, Box<dyn std::error::Error>> {
    use lev_cpu::{matvec_q6k_f32, matvec_q6k_f64, Q6K_BLOCK_BYTES};
    use lev_forge::q6k::{derive, emit_q6k_matvec, group_count};
    let g = GgufFile::open(path)?;
    let t = g.tensor(name).ok_or("tensor not found")?;
    if t.ty != GgmlType::Q6K || t.dims.len() != 2 {
        return Err(format!("{name}: expected a 2-D Q6_K tensor, got {} {:?}", t.ty.name(), t.dims).into());
    }
    let (cols, total_rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let rows = max_rows.min(total_rows);
    let bpr = cols / QK_K;
    let mut file = File::open(path)?;
    let weights = g.read_tensor_bytes(&mut file, t, 0, (rows * bpr * Q6K_BLOCK_BYTES) as u64)?;
    let mut rng = SplitMix64(0x4C45_5649);
    let x: Vec<f32> = (0..cols).map(|_| rng.next_signed_unit()).collect();
    let cpu = matvec_q6k_f32(&weights, rows, cols, &x)?;
    let oracle = matvec_q6k_f64(&weights, rows, cols, &x)?;
    let cpu64: Vec<f64> = cpu.iter().map(|&v| v as f64).collect();

    let ctx = VulkanContext::open(&DeviceSelector::Auto)?;
    let info = ctx.info().clone();
    let cache_dir = std::env::var_os("LEV_CACHE_DIR")
        .map(|p| std::path::PathBuf::from(p).join("spv"))
        .unwrap_or_else(|| std::env::temp_dir().join("lev-forge").join("spv"));
    println!("device [{}] {}", info.index, info.name);
    println!("tensor {name}: Q6_K [{cols} x {total_rows}], {rows} rows, {:.1} MB of weights, gate < {GATE:e}", weights.len() as f64 / 1e6);
    println!("{:<10} {:<6} {:>12} {:>10} {:>10} {:>12} {}", "schedule", "cache", "max_rel_err", "vs_cpu", "median_us", "GB/s", "gate");
    let mut measured: Vec<Option<f64>> = Vec::new();
    let candidates = derive(bpr, &info.caps);
    let mut all_pass = true;
    for s in &candidates {
        let src = emit_q6k_matvec(s)?;
        let (spirv, hit) = compile_glsl_cached(&src, &cache_dir)?;
        let run = ctx.run_matvec(&spirv, group_count(s, rows as u32), Q6K_BLOCK_BYTES, &weights, rows as u32, cols as u32, &x, 3, 11)?;
        let err = max_rel_err(&run.y, &oracle);
        let vs_cpu = max_rel_err(&run.y, &cpu64);
        let pass = err.is_some_and(|e| e < GATE);
        all_pass &= pass;
        let med = run.median_us();
        println!(
            "T={:<3} R={:<2}  {:<6} {:>12} {:>10} {:>10.1} {:>12.1} {}",
            s.t,
            s.r,
            if hit { "hit" } else { "miss" },
            err.map_or("rejected".to_string(), |e| format!("{e:.2e}")),
            vs_cpu.map_or("rejected".to_string(), |e| format!("{e:.1e}")),
            med,
            weights.len() as f64 / (med * 1e-6) / 1e9,
            if pass { "PASS" } else { "FAIL" }
        );
        measured.push(pass.then_some(med));
    }
    let best = measured.iter().enumerate().filter_map(|(i, m)| m.map(|v| (i, v))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    match best {
        Some((i, v)) => println!("selected: T={} R={} ({:.1} us)", candidates[i].t, candidates[i].r, v),
        None => println!("selected: none (no candidate passed the gate)"),
    }
    Ok(all_pass)
}
fn ref_forward(path: &str, ids: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tokens: Vec<usize> = ids.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    let mut m = lev_model::Model::open(path)?;
    println!("config: {:?}", m.cfg);
    let mut logits = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        let start = std::time::Instant::now();
        logits = m.forward(t, None)?;
        println!("token {i} (id {t}): {:.1} s", start.elapsed().as_secs_f64());
    }
    for (id, logit, logprob) in lev_model::log_softmax_top(&logits, 10) {
        println!("id {id:>7}  logit {logit:>9.4}  logprob {logprob:>9.4}");
    }
    Ok(())
}

/// Two-column pass against two single-column passes, on the same tokens.
fn gpu_pair(path: &str, ids: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let tokens = parse_ids(ids)?;
    if tokens.len() < 2 {
        return Err("need at least two tokens".into());
    }
    let mut gpu = lev_model::gpu::GpuModel::open(path, 512, true)?;
    let (head, pair) = tokens.split_at(tokens.len() - 2);
    let use_feed = std::env::var_os("LEV_PAIR_CHUNK").is_some();
    let feed = |gpu: &mut lev_model::gpu::GpuModel| -> Result<(), Box<dyn std::error::Error>> {
        if use_feed {
            gpu.feed(head)?;
        } else {
            for &t in head {
                gpu.forward(t)?;
            }
        }
        Ok(())
    };
    feed(&mut gpu)?;
    let seq = vec![gpu.forward(pair[0])?, gpu.forward(pair[1])?];
    let seq_layers0 = gpu.layer_outputs_col(0).unwrap();
    gpu.reset();
    feed(&mut gpu)?;
    let seq2 = vec![gpu.forward(pair[0])?, gpu.forward(pair[1])?];
    for c in 0..2 {
        let a64: Vec<f64> = seq[c].iter().map(|&v| v as f64).collect();
        println!("repeat run column {c}: max_rel_err {:.2e}", max_rel_err(&seq2[c], &a64).unwrap_or(f64::INFINITY));
    }
    gpu.reset();
    feed(&mut gpu)?;
    let both = gpu.forward_pair(pair[0], pair[1])?;
    let pair_layers = [gpu.layer_outputs_col(0).unwrap(), gpu.layer_outputs_col(1).unwrap()];
    // seq_layers0 holds the residual after the second sequential token (column 0 of a single pass).
    for (l, (a, b)) in seq_layers0.iter().zip(&pair_layers[1]).enumerate() {
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        println!("layer {l:>2}: pair column 1 vs sequential second token max_rel_err {:.2e}", max_rel_err(b, &a64).unwrap_or(f64::INFINITY));
    }
    let mut pass = true;
    for (c, (a, b)) in seq.iter().zip(&both).enumerate() {
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let e = max_rel_err(b, &a64).unwrap_or(f64::INFINITY);
        let same_top = lev_model::argmax(a) == lev_model::argmax(b);
        println!("column {c}: logits max_rel_err {e:.2e}, argmax {}", if same_top { "same" } else { "DIFFERENT" });
        pass &= e < 1e-2 && same_top;
    }
    // The state after a pair must continue correctly: one more token each way.
    let next = lev_model::argmax(&both[1]);
    let l_pair = gpu.forward(next)?;
    gpu.reset();
    for &t in &tokens {
        gpu.forward(t)?;
    }
    let l_seq = gpu.forward(next)?;
    let l64: Vec<f64> = l_seq.iter().map(|&v| v as f64).collect();
    let e = max_rel_err(&l_pair, &l64).unwrap_or(f64::INFINITY);
    println!("continuation after the pair: logits max_rel_err {e:.2e}");
    pass &= e < 1e-2;
    println!("{}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

/// Decodes several prompts together in one ragged batch (one slot each) and compares every
/// sequence with the same prompt decoded alone: `prompts` are token id lists separated by `;`.
fn gpu_batch(path: &str, prompts: &str, n: &str) -> Result<bool, Box<dyn std::error::Error>> {
    use lev_model::gpu::{BatchItem, GpuModel};
    let n: usize = n.parse()?;
    let prompts: Vec<Vec<usize>> = prompts.split(';').map(parse_ids).collect::<Result<_, _>>()?;
    let b = prompts.len();
    std::env::set_var("LEV_BATCH_SLOTS", b.to_string());
    let ctx: usize = std::env::var("LEV_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let mut gpu = GpuModel::open(path, ctx, false)?;

    // Each prompt alone (with the logit margin of every token it chose).
    let mut solo: Vec<Vec<usize>> = Vec::new();
    let mut margins: Vec<Vec<f32>> = Vec::new();
    let mut solo_secs = 0.0;
    let mut seen: Vec<(Vec<usize>, usize)> = Vec::new(); // distinct prompt -> index of its solo run
    for p in &prompts {
        if let Some((_, at)) = seen.iter().find(|(q, _)| q == p) {
            let (o, m) = (solo[*at].clone(), margins[*at].clone());
            solo.push(o);
            margins.push(m);
            continue;
        }
        gpu.reset();
        let (out, st) = gpu.generate(p, n, false)?;
        solo_secs += st.seconds - st.prefill_seconds;
        seen.push((p.clone(), solo.len()));
        solo.push(out);
        margins.push(st.margins);
    }
    let solo_runs = seen.len();

    // Prefill each into the main engine, import into its slot.
    let mut pos: Vec<usize> = Vec::new();
    // LEV_BATCH_REVERSE=1 gives the first prompt the last slot (to tell column and slot effects apart).
    let slot_of = |i: usize| if std::env::var_os("LEV_BATCH_REVERSE").is_some() { b - 1 - i } else { i };
    for (i, p) in prompts.iter().enumerate() {
        gpu.prefill_prompt(p)?;
        gpu.batch_import(slot_of(i))?;
        pos.push(p.len());
    }

    // Teacher forcing: every step feeds the solo run's token, so a near tie cannot compound. The
    // batch output at step k must be the solo token k + 1 unless that choice was a near tie.
    let t0 = std::time::Instant::now();
    let mut gpu_us = 0.0;
    let (mut total, mut equal, mut worst_tie) = (0usize, 0usize, 0f32);
    let mut unexplained = 0usize;
    for k in 0..n - 1 {
        let items: Vec<BatchItem> = (0..b).map(|i| BatchItem { slot: slot_of(i), token: solo[i][k], pos: pos[i] + k }).collect();
        let res = gpu.batch_step(&items)?;
        gpu_us += gpu.last_gpu_us;
        if k == 0 && std::env::var_os("LEV_BATCH_DEBUG").is_some() {
            // Identical sequences must give identical columns: where does column 16 differ from column 0?
            for (name, v) in gpu.debug_buffers()? {
                let stride = v.len() / lev_model::gpu::DEBUG_COLS;
                if stride == 0 {
                    continue;
                }
                let col = |c: usize| &v[c * stride..(c + 1) * stride];
                let d = |a: usize, b: usize| col(a).iter().zip(col(b)).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                println!("debug {name:<9} column 0 vs 1: {:.3e}   vs 15: {:.3e}   vs 16: {:.3e}   vs 17: {:.3e}   (16 vs 17: {:.3e}, 16 vs 18: {:.3e})", d(0, 1), d(0, 15), d(0, 16), d(0, 17), d(16, 17), d(16, 18));
            }
        }
        for i in 0..b {
            total += 1;
            if res[i].0 == solo[i][k + 1] {
                equal += 1;
            } else {
                // The solo margin of token k + 1 is how far the best choice led the second best.
                let m = margins[i][k + 1];
                worst_tie = worst_tie.max(m);
                if m > 0.1 {
                    unexplained += 1;
                }
                println!("sequence {i}, step {k}: batch {} solo {} (solo margin {m:.3}, batch margin {:.3})", res[i].0, solo[i][k + 1], res[i].1);
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("{equal} of {total} next tokens equal the solo run; the {} differing ones are near ties (largest solo margin {worst_tie:.3})", total - equal);
    let steps = (n - 1) as f64;
    println!("batch of {b}: {:.1} aggregate tok/s ({:.1} tok/s per sequence), {:.1} ms per step ({:.1} ms GPU)", b as f64 * steps / secs, steps / secs, secs / steps * 1e3, gpu_us / steps / 1e3);
    println!("solo: {:.1} tok/s per sequence", steps * solo_runs as f64 / solo_secs.max(1e-9));
    let ok = unexplained == 0;
    println!("{}", if ok { "PASS" } else { "FAIL: differences where the solo run was not a near tie" });
    Ok(ok)
}
/// Runs a fixture (ir.json, weights.pack, inputs.pack, expected.pack from tools/model_ir/make_fixture.py)
/// through the IR interpreter and compares every output with the PyTorch result.
fn ir_run(dir: &str) -> Result<bool, Box<dyn std::error::Error>> {
    use lev_ir::{interp, ir::Graph, pack};
    let d = std::path::Path::new(dir);
    let g = Graph::parse(&std::fs::read_to_string(d.join("ir.json"))?)?;
    g.validate()?;
    let unsupported: Vec<_> = g.op_counts().into_iter().filter(|(op, _)| !interp::is_supported(op)).collect();
    if !unsupported.is_empty() {
        return Err(format!("unsupported operators: {unsupported:?}").into());
    }
    let params = pack::read(&std::fs::read(d.join("weights.pack"))?)?;
    let inputs = pack::read(&std::fs::read(d.join("inputs.pack"))?)?;
    let expected = pack::read(&std::fs::read(d.join("expected.pack"))?)?;
    let t0 = std::time::Instant::now();
    let outs = interp::run(&g, &params, &inputs)?;
    println!("interpreted {} nodes in {:.1} ms", g.nodes.len(), t0.elapsed().as_secs_f64() * 1e3);
    let mut pass = outs.len() == expected.len();
    for (i, o) in outs.iter().enumerate() {
        let want = match expected.get(&format!("out_{i}")) {
            Some(pack::Packed::F32(t)) => t,
            _ => return Err(format!("expected.pack has no float tensor out_{i}").into()),
        };
        let got = match &**o {
            interp::V::T(t) => t,
            _ => return Err(format!("output {i} is not a float tensor").into()),
        };
        if got.shape != want.shape {
            println!("out_{i}: shape {:?} != {:?}", got.shape, want.shape);
            pass = false;
            continue;
        }
        let (mut max_abs, mut max_ref) = (0f32, 0f32);
        for (a, b) in got.data.iter().zip(&want.data) {
            max_abs = max_abs.max((a - b).abs());
            max_ref = max_ref.max(b.abs());
        }
        let ok = max_abs <= 1e-4 + 1e-4 * max_ref;
        pass &= ok;
        println!("out_{i}: shape {:?} max_abs_err {max_abs:.2e} (max |ref| {max_ref:.2e}) {}", got.shape, if ok { "ok" } else { "FAIL" });
    }
    println!("{}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

/// Parameters of an IR graph read from the GGUF file (dequantized on first use and kept).
struct GgufParams {
    gguf: lev_gguf::GgufFile,
    file: std::cell::RefCell<std::fs::File>,
    shapes: std::collections::BTreeMap<String, Vec<usize>>,
    cache: std::cell::RefCell<std::collections::BTreeMap<String, std::rc::Rc<lev_ir::interp::V>>>,
}

impl lev_ir::interp::ParamSource for GgufParams {
    fn param(&self, name: &str) -> Result<std::rc::Rc<lev_ir::interp::V>, String> {
        if let Some(v) = self.cache.borrow().get(name) {
            return Ok(v.clone());
        }
        let (gname, _) = lev_ir::bind::to_gguf(name).ok_or(format!("no GGUF binding for {name}"))?;
        let shape = self.shapes.get(name).cloned().ok_or(format!("{name} is not a parameter of the graph"))?;
        let data = lev_model::weights::dequantize(&self.gguf, &mut self.file.borrow_mut(), &gname, 0, None).map_err(|e| e.to_string())?;
        if data.len() != shape.iter().product::<usize>() {
            return Err(format!("{name}: {} elements, expected {shape:?}", data.len()));
        }
        let v = std::rc::Rc::new(lev_ir::interp::V::T(lev_ir::tensor::Tensor::new(shape, data)));
        self.cache.borrow_mut().insert(name.to_string(), v.clone());
        Ok(v)
    }
}

/// Runs the first `layers` layers of the retargeted IR on GGUF weights (dequantized to f32) for a
/// prompt, and compares the residual stream after the last of them (last token) with the CPU
/// reference `Model`, which matches llama.cpp.
fn ir_gguf(ir_path: &str, gguf_path: &str, ids: &str, layers: &str) -> Result<bool, Box<dyn std::error::Error>> {
    use lev_ir::interp::{self, RunOptions, V};
    use lev_ir::ir::{Arg, Graph};
    use lev_ir::pack::Packed;
    use lev_ir::tensor::{IntTensor, Tensor};
    use std::rc::Rc;

    let tokens = parse_ids(ids)?;
    let n_layers: usize = layers.parse()?;
    let t = tokens.len();
    let raw = Graph::parse(&std::fs::read_to_string(ir_path)?)?;
    let g = lev_ir::pass::retarget_gguf(&raw)?;
    g.validate()?;
    let gguf = lev_gguf::GgufFile::open(gguf_path)?;
    let cfg = lev_model::Config::from_gguf(&gguf)?;
    let shapes = g.params.iter().map(|p| (p.name.clone(), p.shape.iter().map(|d| d.parse::<usize>().unwrap_or(0)).collect())).collect();
    let params = GgufParams { gguf, file: std::cell::RefCell::new(std::fs::File::open(gguf_path)?), shapes, cache: Default::default() };

    // Where layer `n_layers - 1` ends: just before the first node that reads layer `n_layers`.
    fn mentions(a: &Arg, prefix: &str) -> bool {
        match a {
            Arg::Param(p) => p.starts_with(prefix),
            Arg::List(l) => l.iter().any(|x| mentions(x, prefix)),
            _ => false,
        }
    }
    let next = format!("model.layers.{n_layers}.");
    let stop = g.nodes.iter().position(|n| n.args.iter().any(|a| mentions(a, &next))).map(|i| i - 1).ok_or("the graph has no such layer boundary")?;

    // Inputs: token ids, rope tables, zero states.
    let rd = cfg.rope_dim;
    let (mut cos, mut sin) = (vec![0f32; t * rd], vec![0f32; t * rd]);
    for p in 0..t {
        for i in 0..rd {
            let f = cfg.rope_base.powf(-2.0 * (i % (rd / 2)) as f64 / rd as f64);
            cos[p * rd + i] = (p as f64 * f).cos() as f32;
            sin[p * rd + i] = (p as f64 * f).sin() as f32;
        }
    }
    let mut inputs = std::collections::BTreeMap::new();
    inputs.insert("token_ids".to_string(), Packed::I64(IntTensor { shape: vec![t], data: tokens.iter().map(|&x| x as i64).collect() }));
    inputs.insert("cos".to_string(), Packed::F32(Tensor::new(vec![t, rd], cos)));
    inputs.insert("sin".to_string(), Packed::F32(Tensor::new(vec![t, rd], sin)));
    for v in g.inputs.iter().filter(|v| v.name.starts_with("states_")) {
        let shape: Vec<usize> = v.shape.iter().map(|d| d.parse::<usize>().unwrap_or(0)).collect();
        inputs.insert(v.name.clone(), Packed::F32(Tensor::zeros(shape)));
    }

    // The embedding table is too large to dequantize whole: gather the rows of the prompt.
    let emb_node = g.nodes.iter().position(|n| n.op == "aten::embedding.default").ok_or("no embedding node")?;
    let mut emb = Vec::with_capacity(t * cfg.n_embd);
    {
        let mut f = params.file.borrow_mut();
        for &id in &tokens {
            emb.extend(lev_model::weights::dequantize(&params.gguf, &mut f, "token_embd.weight", id, Some(1))?);
        }
    }
    let mut opts = RunOptions { stop_after: Some(stop), ..Default::default() };
    opts.overrides.insert(emb_node, Rc::new(V::T(Tensor::new(vec![t, cfg.n_embd], emb))));

    let t0 = std::time::Instant::now();
    let vals = interp::run_nodes(&g, &params, &inputs, &opts)?;
    println!("IR interpreter: {} layers, {t} tokens, {:.1} s", n_layers, t0.elapsed().as_secs_f64());
    let V::T(x) = &**vals[stop].as_ref().ok_or("no value at the stop node")? else { return Err("stop node is not a tensor".into()) };
    let got = &x.data[(t - 1) * cfg.n_embd..t * cfg.n_embd];

    let t0 = std::time::Instant::now();
    let mut cpu = lev_model::Model::open(gguf_path)?;
    let mut trace = lev_model::Trace::default();
    for &tok in &tokens {
        trace = lev_model::Trace::default();
        cpu.forward(tok, Some(&mut trace))?;
    }
    println!("CPU reference: {:.1} s", t0.elapsed().as_secs_f64());
    let want = &trace.layer_out[n_layers - 1];
    let max_ref = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let (mut max_abs, mut at) = (0f32, 0usize);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        if (a - b).abs() > max_abs {
            max_abs = (a - b).abs();
            at = i;
        }
    }
    let rel = max_abs / max_ref;
    println!("residual after layer {}: max_abs_err {max_abs:.3e} at {at} (max |ref| {max_ref:.3e}) relative {rel:.2e}", n_layers - 1);
    let ok = rel < 2e-3;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(ok)
}

/// Prints the nodes of one layer of the retargeted IR (operator and arguments).
fn ir_dump(ir_path: &str, layer: &str) -> Result<(), Box<dyn std::error::Error>> {
    use lev_ir::ir::{Arg, Graph};
    let g = lev_ir::pass::retarget_gguf(&Graph::parse(&std::fs::read_to_string(ir_path)?)?)?;
    let l: usize = layer.parse()?;
    let (cur, next) = (format!("model.layers.{l}."), format!("model.layers.{}.", l + 1));
    fn mentions(a: &Arg, prefix: &str) -> bool {
        match a {
            Arg::Param(p) => p.starts_with(prefix),
            Arg::List(v) => v.iter().any(|x| mentions(x, prefix)),
            _ => false,
        }
    }
    let first = g.nodes.iter().position(|n| n.args.iter().any(|a| mentions(a, &cur))).ok_or("no such layer")?;
    let last = g.nodes.iter().position(|n| n.args.iter().any(|a| mentions(a, &next))).unwrap_or(g.nodes.len());
    fn show(a: &Arg, prefix: &str) -> String {
        match a {
            Arg::Node(n) => format!("%{n}"),
            Arg::Param(p) => p.strip_prefix(prefix).map(|s| format!("@{s}")).unwrap_or_else(|| format!("@{p}")),
            Arg::Input(i) => format!("${i}"),
            Arg::Int(i) => i.to_string(),
            Arg::Float(f) => f.to_string(),
            Arg::Bool(b) => b.to_string(),
            Arg::List(v) => format!("[{}]", v.iter().map(|x| show(x, prefix)).collect::<Vec<_>>().join(", ")),
            other => format!("{other:?}"),
        }
    }
    for (i, n) in g.nodes.iter().enumerate().take(last).skip(first) {
        println!("%{i} = {}({})", n.op, n.args.iter().map(|a| show(a, &cur)).collect::<Vec<_>>().join(", "));
    }
    Ok(())
}

/// Loads an IR file, validates it and checks that every parameter binds to a GGUF tensor of the
/// same size.
fn ir_check(ir_path: &str, gguf_path: &str) -> Result<bool, Box<dyn std::error::Error>> {
    use lev_ir::{bind, interp, ir::Graph};
    let g = Graph::parse(&std::fs::read_to_string(ir_path)?)?;
    g.validate()?;
    println!("{} nodes, {} parameters, {} inputs, {} outputs", g.nodes.len(), g.params.len(), g.inputs.len(), g.outputs.len());
    let mut pass = true;
    for (op, c) in g.op_counts() {
        let sup = interp::is_supported(&op);
        pass &= sup;
        println!("  {c:5}  {op}{}", if sup { "" } else { "   <- not supported by the interpreter" });
    }
    let gguf = lev_gguf::GgufFile::open(gguf_path)?;
    let (mut bound, mut transformed) = (0usize, 0usize);
    let mut used = std::collections::BTreeSet::new();
    for p in &g.params {
        let Some((name, t)) = bind::to_gguf(&p.name) else {
            println!("no GGUF binding rule for {}", p.name);
            pass = false;
            continue;
        };
        let Some(info) = gguf.tensor(&name) else {
            println!("{}: GGUF tensor {name} not found", p.name);
            pass = false;
            continue;
        };
        let ir_elems: u64 = p.shape.iter().map(|d| d.parse::<u64>().unwrap_or(0)).product();
        let gg_elems = info.element_count().unwrap_or(0);
        if ir_elems != gg_elems {
            println!("{}: {} elements in the IR, {gg_elems} in GGUF tensor {name}", p.name, ir_elems);
            pass = false;
            continue;
        }
        // 2-D weights: GGUF lists the dimensions in the opposite order.
        if p.shape.len() == 2 {
            let (a, b): (u64, u64) = (p.shape[0].parse().unwrap_or(0), p.shape[1].parse().unwrap_or(0));
            if info.dims != [b, a] {
                println!("{}: dimensions {:?} do not match GGUF {:?} of {name}", p.name, p.shape, info.dims);
                pass = false;
                continue;
            }
        }
        bound += 1;
        transformed += (t != bind::Transform::Plain) as usize;
        used.insert(name);
    }
    // GGUF tensors that no IR parameter claims (the MTP block, for instance).
    let unclaimed: Vec<_> = gguf.tensors.iter().filter(|t| !used.contains(&t.name)).map(|t| t.name.as_str()).collect();
    println!("{bound} of {} parameters bound ({transformed} with a stored transform); {} GGUF tensors not used by the IR", g.params.len(), unclaimed.len());
    for n in unclaimed.iter().take(12) {
        println!("  unclaimed: {n}");
    }
    println!("{}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

/// Three-column pass against single-token passes on the last three tokens, for each number of
/// kept columns (the state after that many must continue correctly through another three-column pass).
fn gpu_triple(path: &str, ids: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let tokens = parse_ids(ids)?;
    if tokens.len() < 6 {
        return Err("need at least six tokens".into());
    }
    let mut gpu = lev_model::gpu::GpuModel::open(path, 512, true)?;
    let n = tokens.len();
    let (head, rest) = tokens.split_at(n - 6);
    let (t3, t3b) = (&rest[..3], &rest[3..]);
    for &t in head {
        gpu.forward(t)?;
    }
    let mut seq = Vec::new();
    for &t in &rest[..3] {
        seq.push(gpu.forward(t)?);
    }
    let seq_layers = gpu.layer_outputs_col(0).unwrap();
    gpu.reset();
    for &t in head {
        gpu.forward(t)?;
    }
    let both = gpu.forward_triple([t3[0], t3[1], t3[2]], 3)?;
    let tri_layers = [gpu.layer_outputs_col(0).unwrap(), gpu.layer_outputs_col(1).unwrap(), gpu.layer_outputs_col(2).unwrap()];
    let mut worst = 0f64;
    for (l, (a, b)) in seq_layers.iter().zip(&tri_layers[2]).enumerate() {
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let e = max_rel_err(b, &a64).unwrap_or(f64::INFINITY);
        worst = worst.max(e);
        if l % 8 == 7 || l < 2 {
            println!("layer {l:>2}: triple column 2 vs sequential third token max_rel_err {e:.2e}");
        }
    }
    println!("worst layer error of column 2: {worst:.2e}");
    let mut pass = true;
    for (c, (a, b)) in seq.iter().zip(&both).enumerate() {
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        let e = max_rel_err(b, &a64).unwrap_or(f64::INFINITY);
        let same_top = lev_model::argmax(a) == lev_model::argmax(b);
        println!("column {c}: logits max_rel_err {e:.2e}, argmax {}", if same_top { "same" } else { "DIFFERENT" });
        // Logit errors of any two valid schedules sit at the int8 noise level of the input (see gpu-pair);
        // the criterion is that every greedy choice agrees.
        pass &= same_top;
    }
    // Every number of kept columns: state after k columns, then another three-column pass.
    for keep in 1..=3usize {
        gpu.reset();
        for &t in head {
            gpu.forward(t)?;
        }
        gpu.forward_triple([t3[0], t3[1], t3[2]], keep)?;
        // The tokens after the kept ones: the remaining original tokens, then the next block.
        let follow: Vec<usize> = rest[keep..].to_vec();
        let cont = gpu.forward_triple([follow[0], follow[1], follow[2]], 3)?;
        gpu.reset();
        for &t in head {
            gpu.forward(t)?;
        }
        for &t in &rest[..keep] {
            gpu.forward(t)?;
        }
        let mut want = Vec::new();
        for &t in &follow[..3] {
            want.push(gpu.forward(t)?);
        }
        for c in 0..3 {
            let a64: Vec<f64> = want[c].iter().map(|&v| v as f64).collect();
            let e = max_rel_err(&cont[c], &a64).unwrap_or(f64::INFINITY);
            let same_top = lev_model::argmax(&want[c]) == lev_model::argmax(&cont[c]);
            println!("keep {keep}, continuation column {c}: logits max_rel_err {e:.2e}, argmax {}", if same_top { "same" } else { "DIFFERENT" });
            pass &= same_top;
        }
    }
    let _ = t3b;
    println!("{}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}

/// Plain greedy decoding against MTP speculative decoding on the same prompt.
fn gpu_spec(path: &str, ids: &str, n: usize) -> Result<bool, Box<dyn std::error::Error>> {
    let prompt = parse_ids(ids)?;
    let mut gpu = lev_model::gpu::GpuModel::open(path, std::env::var("LEV_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(1024), false)?;
    println!("device {}", gpu.device_name());
    let (plain, ps) = gpu.generate(&prompt, n, false)?;
    gpu.reset();
    let (spec, ss) = gpu.generate(&prompt, n, true)?;
    println!("plain: {} tokens in {:.2} s (GPU {:.1} ms)", ps.tokens, ps.seconds, ps.main_us / 1000.0);
    println!("prefill: {} prompt tokens in {:.2} s = {:.1} tok/s", prompt.len(), ps.prefill_seconds, prompt.len() as f64 / ps.prefill_seconds);
    println!("spec : {} tokens in {:.2} s, {} verify steps, {} drafts accepted ({:.2} per step)", ss.tokens, ss.seconds, ss.steps, ss.accepted, ss.accepted as f64 / ss.steps.max(1) as f64);
    println!("spec GPU time: main {:.1} ms + mtp {:.1} ms; host/other {:.1} ms", ss.main_us / 1000.0, ss.mtp_us / 1000.0, ss.seconds * 1000.0 - (ss.main_us + ss.mtp_us) / 1000.0);
    println!("plain host/other {:.1} ms", ps.seconds * 1000.0 - ps.main_us / 1000.0);
    let dec = |st: &lev_model::gpu::SpecStats| (st.tokens.saturating_sub(1)) as f64 / (st.seconds - st.prefill_seconds);
    println!("decode-only rate (prompt excluded): plain {:.2} tok/s, spec {:.2} tok/s", dec(&ps), dec(&ss));
    println!("speedup (wall, prompt included in both): {:.2}x", ps.seconds / ss.seconds);
    let first_diff = plain.iter().zip(&spec).position(|(a, b)| a != b);
    match first_diff {
        None => println!("outputs identical ({} tokens)", plain.len()),
        Some(i) => println!(
            "outputs DIVERGE at token {i}: plain {:?} vs spec {:?}; plain logit margin at that token {:.3} (token {} margin {:.3})",
            &plain[i..(i + 4).min(plain.len())],
            &spec[i..(i + 4).min(spec.len())],
            ps.margins.get(i).copied().unwrap_or(f32::NAN),
            i.saturating_sub(1),
            ps.margins.get(i.saturating_sub(1)).copied().unwrap_or(f32::NAN)
        ),
    }
    println!("ids: {plain:?}");
    Ok(first_diff.is_none())
}
fn gpu_profile(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = lev_model::gpu::GpuModel::open_with(path, 512, false, true)?;
    let tokens = [760usize, 6511, 314, 9338, 369, 11751, 13, 198];
    let mut last = Vec::new();
    for &t in &tokens {
        last = gpu.profile_token(t)?;
    }
    let single_labels = gpu.labels().to_vec();
    let mut pair = Vec::new();
    for _ in 0..3 {
        pair = gpu.profile_pair(11751, 13)?;
    }
    let pair_labels = gpu.labels_pair().to_vec();
    let mut chunk = Vec::new();
    for _ in 0..3 {
        chunk = gpu.profile_chunk(&[760, 6511, 314, 9338, 369, 11751, 13, 198])?;
    }
    let chunk_labels = gpu.labels_prefill().to_vec();
    let mut big = Vec::new();
    let toks64: Vec<usize> = (0..64).map(|i| [760usize, 6511, 314, 9338, 369, 11751, 13, 198][i % 8]).collect();
    for _ in 0..3 {
        big = gpu.profile_big(&toks64)?;
    }
    let big_labels = gpu.labels_big().to_vec();
    for (title, labels, times) in [("single-token pass", &single_labels, &last), ("two-token pass", &pair_labels, &pair), ("8-token prefill chunk", &chunk_labels, &chunk), ("64-token matrix-core chunk", &big_labels, &big)] {
        let total: f64 = times.iter().sum();
        let mut by: std::collections::BTreeMap<String, (usize, f64)> = std::collections::BTreeMap::new();
        for (l, us) in labels.iter().zip(times.iter()) {
            let e = by.entry(l.clone()).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += us;
        }
        let mut rows: Vec<_> = by.into_iter().collect();
        rows.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        println!("== {title}: total {:.2} ms over {} dispatches", total / 1000.0, times.len());
        println!("{:<22} {:>5} {:>10} {:>8} {:>10}", "step kind", "count", "total_ms", "share", "avg_us");
        for (k, (n, us)) in rows.into_iter().take(14) {
            println!("{:<22} {:>5} {:>10.2} {:>7.1}% {:>10.1}", k, n, us / 1000.0, us / total * 100.0, us / n as f64);
        }
    }
    Ok(())
}
/// Final-token top logits of the GPU path (chunked prefill unless LEV_NO_CHUNK is set).
fn gpu_top(path: &str, ids: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = parse_ids(ids)?;
    let mut gpu = lev_model::gpu::GpuModel::open(path, 2048, false)?;
    let (head, last) = tokens.split_at(tokens.len() - 1);
    if std::env::var_os("LEV_NO_CHUNK").is_some() {
        for &t in head {
            gpu.forward(t)?;
        }
    } else {
        gpu.feed(head)?;
    }
    let logits = gpu.forward(last[0])?;
    for (id, logit, logprob) in lev_model::log_softmax_top(&logits, 10) {
        println!("id {id:>7}  logit {logit:>9.4}  logprob {logprob:>9.4}");
    }
    Ok(())
}

/// Cooperative-matrix smoke test and throughput: 1024^3 f16 GEMM against an f64 reference.
fn vk_coopmat() -> Result<bool, Box<dyn std::error::Error>> {
    let ctx = VulkanContext::open(&DeviceSelector::Auto)?;
    if !ctx.info().coopmat {
        return Err("cooperative matrix is not enabled on this device".into());
    }
    let (m, n, k) = (1024usize, 1024usize, 1024usize);
    let mut rng = SplitMix64(0x434F_4F50);
    let a: Vec<u16> = (0..m * k).map(|_| lev_cpu::f32_to_f16_bits(rng.next_signed_unit())).collect();
    let b: Vec<u16> = (0..n * k).map(|_| lev_cpu::f32_to_f16_bits(rng.next_signed_unit())).collect();
    let as_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (abuf, bbuf) = (ctx.create_device_buffer((m * k * 2) as u64)?, ctx.create_device_buffer((n * k * 2) as u64)?);
    ctx.upload(&abuf, 0, &as_bytes(&a))?;
    ctx.upload(&bbuf, 0, &as_bytes(&b))?;
    let cbuf = ctx.create_buffer((m * n * 4) as u64)?;
    let cache_dir = std::env::var_os("LEV_CACHE_DIR").map(|p| std::path::PathBuf::from(p).join("spv")).unwrap_or_else(|| std::env::temp_dir().join("lev-forge").join("spv"));
    let (spv, _) = compile_glsl_cached(&lev_forge::coopmat::emit_f16_gemm_test(), &cache_dir)?;
    let kernel = ctx.create_kernel(&spv, 3, 12)?;
    let push = [(m as u32).to_le_bytes(), (n as u32).to_le_bytes(), (k as u32).to_le_bytes()].concat();
    let steps = vec![lev_vulkan::Step { kernel: &kernel, bindings: vec![(&abuf, 0, abuf.size()), (&bbuf, 0, bbuf.size()), (&cbuf, 0, cbuf.size())], push, groups: ((m / 16) * (n / 16)) as u32 }];
    let prog = ctx.build_program(&steps, false)?;
    let mut times = Vec::new();
    for _ in 0..3 {
        prog.run()?;
    }
    for _ in 0..9 {
        times.push(prog.run()?);
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = times[times.len() / 2];
    let c = cbuf.read_f32(0, m * n);
    // Reference on a sample of outputs (f64 from the same f16 inputs).
    let mut worst = 0f64;
    for t in 0..2000usize {
        let (i, j) = ((t * 7919) % m, (t * 104729) % n);
        let mut acc = 0f64;
        for kk in 0..k {
            acc += lev_cpu::f16_to_f64(a[i * k + kk]) * lev_cpu::f16_to_f64(b[j * k + kk]);
        }
        worst = worst.max((c[i * n + j] as f64 - acc).abs() / acc.abs().max(1.0));
    }
    let flops = 2.0 * (m * n * k) as f64;
    println!("f16 gemm {m}x{n}x{k}: median {:.1} us = {:.2} TFLOP/s, max rel err on 2000 samples {:.2e}", med, flops / (med * 1e-6) / 1e12, worst);
    Ok(worst < 1e-3)
}

/// Matrix-core Q4_K GEMM over 64 tokens against an f64 oracle that rounds the
/// weights and activations to f16 exactly as the kernel does.
fn vk_gemm(path: &str, max_rows: usize, name: &str) -> Result<bool, Box<dyn std::error::Error>> {
    use lev_forge::coopmat;
    let g = GgufFile::open(path)?;
    let t = g.tensor(name).ok_or("tensor not found")?;
    let is_q6 = t.ty == GgmlType::Q6K;
    let block_bytes = if is_q6 { lev_cpu::Q6K_BLOCK_BYTES } else { Q4K_BLOCK_BYTES };
    let (cols, total_rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let rows = (max_rows.min(total_rows) / 32) * 32;
    let bpr = cols / QK_K;
    let ntok: usize = std::env::var("LEV_GEMM_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(coopmat::GEMM_TOKENS);
    let mut file = File::open(path)?;
    let weights = g.read_tensor_bytes(&mut file, t, 0, (rows * bpr * block_bytes) as u64)?;
    let mut rng = SplitMix64(0x4745_4D4D);
    let x: Vec<f32> = (0..ntok * cols).map(|_| rng.next_signed_unit()).collect();
    // Oracle: f16-rounded weights and activations, f64 accumulation.
    let xr: Vec<f64> = x.iter().map(|&v| lev_cpu::f16_to_f64(lev_cpu::f32_to_f16_bits(v))).collect();
    let mut wr = vec![0f64; rows * cols];
    let mut tmp = [0f32; QK_K];
    for r in 0..rows {
        for b in 0..bpr {
            let off = (r * bpr + b) * block_bytes;
            if is_q6 {
                lev_cpu::dequant_q6k_block_f32(&weights[off..off + block_bytes], &mut tmp);
            } else {
                lev_cpu::dequant_block_f32(&weights[off..off + block_bytes], &mut tmp);
            }
            for (i, v) in tmp.iter().enumerate() {
                wr[r * cols + b * QK_K + i] = lev_cpu::f16_to_f64(lev_cpu::f32_to_f16_bits(*v));
            }
        }
    }
    let ctx = VulkanContext::open(&DeviceSelector::Auto)?;
    if !ctx.info().coopmat {
        return Err("cooperative matrix is not enabled on this device".into());
    }
    let cache_dir = std::env::var_os("LEV_CACHE_DIR").map(|p| std::path::PathBuf::from(p).join("spv")).unwrap_or_else(|| std::env::temp_dir().join("lev-forge").join("spv"));
    let wbuf = ctx.create_device_buffer(weights.len() as u64 + 16)?;
    ctx.upload(&wbuf, 0, &weights)?;
    let xbuf = ctx.create_buffer((x.len() * 4) as u64)?;
    xbuf.write_f32(0, &x);
    let xh = ctx.create_device_buffer((x.len() * 2) as u64)?;
    let ybuf = ctx.create_buffer((ntok * rows * 4) as u64)?;
    let (cspv, _) = compile_glsl_cached(&coopmat::emit_act_f16(), &cache_dir)?;
    let (gspv, hit) = compile_glsl_cached(&{
        let wave: usize = std::env::var("LEV_WAVE").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        // LEV_GEMM_PLAIN selects the older per-32-element kernels for comparison.
        let plain = std::env::var_os("LEV_GEMM_PLAIN").is_some();
        match (is_q6, plain) {
            (true, false) => coopmat::emit_q6k_gemm_blocked_tokens(ntok),
            (true, true) => coopmat::emit_q6k_gemm(wave),
            (false, false) => coopmat::emit_q4k_gemm_blocked_tokens(ntok),
            (false, true) => coopmat::emit_q4k_gemm(wave),
        }
    }, &cache_dir)?;
    let wave: u32 = std::env::var("LEV_WAVE").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let (kc, kg) = (ctx.create_kernel(&cspv, 2, 0)?, ctx.create_kernel_sub(&gspv, 3, 8, Some(wave))?);
    if std::env::var_os("LEV_PIPELINE_INFO").is_some() {
        println!("driver statistics of the gemm kernel:");
        let dump = std::env::var_os("LEV_ISA_DIR").map(std::path::PathBuf::from);
        for l in ctx.kernel_report(&kg, "gemm", dump.as_deref())? {
            println!("{l}");
        }
    }
    let push = [(rows as u32).to_le_bytes(), (bpr as u32).to_le_bytes()].concat();
    let steps = vec![
        lev_vulkan::Step { kernel: &kc, bindings: vec![(&xbuf, 0, xbuf.size()), (&xh, 0, xh.size())], push: Vec::new(), groups: (x.len() / 256) as u32 },
        lev_vulkan::Step { kernel: &kg, bindings: vec![(&wbuf, 0, wbuf.size()), (&xh, 0, xh.size()), (&ybuf, 0, ybuf.size())], push, groups: (rows / 32) as u32 },
    ];
    let prog = ctx.build_program(&steps, false)?;
    let mut times = Vec::new();
    for _ in 0..3 {
        prog.run()?;
    }
    for _ in 0..9 {
        times.push(prog.run()?);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let y = ybuf.read_f32(0, ntok * rows);
    let mut worst = 0f64;
    for tk in 0..ntok {
        for r in (0..rows).step_by(13) {
            let mut acc = 0f64;
            for c in 0..cols {
                acc += wr[r * cols + c] * xr[tk * cols + c];
            }
            worst = worst.max((y[tk * rows + r] as f64 - acc).abs() / acc.abs().max(1.0));
        }
    }
    let flops = 2.0 * (ntok * rows * cols) as f64;
    println!("{} gemm {rows}x{cols}, {ntok} tokens ({}): median {:.1} us (incl. f16 conversion) = {:.2} TFLOP/s, weights {:.1} MB -> {:.1} GB/s, max rel err {:.2e}", t.ty.name(), if hit { "cache hit" } else { "compiled" }, med, flops / (med * 1e-6) / 1e12, weights.len() as f64 / 1e6, weights.len() as f64 / (med * 1e-6) / 1e9, worst);
    Ok(worst < 1e-3)
}

/// Equivalence test of the chunk gated-delta kernel: 8 columns in one dispatch
/// against the single-column kernel applied 8 times, same random inputs and
/// initial state; compares outputs and the final state.
fn vk_delta() -> Result<bool, Box<dyn std::error::Error>> {
    use lev_forge::ops;
    const KH: usize = 16;
    const VH: usize = 32;
    const HS: usize = 128;
    const CD: usize = 8192;
    const NB: usize = 8;
    let ctx = VulkanContext::open(&DeviceSelector::Auto)?;
    let cache_dir = std::env::var_os("LEV_CACHE_DIR").map(|p| std::path::PathBuf::from(p).join("spv")).unwrap_or_else(|| std::env::temp_dir().join("lev-forge").join("spv"));
    let mk = |src: &str| -> Result<lev_vulkan::Kernel, Box<dyn std::error::Error>> {
        let (spv, _) = compile_glsl_cached(src, &cache_dir)?;
        Ok(ctx.create_kernel(&spv, 12, 0)?)
    };
    let k1 = mk(&ops::ssm_delta(KH, VH, HS, 1e-6, CD, 1))?;
    let kc = mk(&ops::ssm_delta_chunk(KH, VH, HS, 1e-6, CD, NB))?;
    let mut rng = SplitMix64(0x44454C54);
    let mut rnd = |n: usize, scale: f32, off: f32| -> Vec<f32> { (0..n).map(|_| rng.next_signed_unit() * scale + off).collect() };
    let qk = rnd(NB * 2 * KH * HS, 0.1, 0.0);
    let conv = rnd(NB * CD, 1.0, 0.0);
    let alpha = rnd(NB * VH, 1.0, 0.0);
    let beta = rnd(NB * VH, 1.0, 0.0);
    let dt = rnd(VH, 0.5, 0.5);
    let a = rnd(VH, 0.4, -0.6);
    let z = rnd(NB * VH * HS, 1.0, 0.0);
    let nw = rnd(HS, 0.3, 1.0);
    let state0 = rnd(VH * HS * HS, 0.05, 0.0);
    let host = |v: &[f32]| -> Result<lev_vulkan::Buf, Box<dyn std::error::Error>> {
        let b = ctx.create_buffer((v.len() * 4) as u64)?;
        b.write_f32(0, v);
        Ok(b)
    };
    let ctrl = ctx.create_buffer(256)?;
    ctrl.zero();
    let (qkb, cvb, dtb, ab, nwb) = (host(&qk)?, host(&conv)?, host(&dt)?, host(&a)?, host(&nw)?);
    let (alb, beb, zb) = (host(&alpha)?, host(&beta)?, host(&z)?);
    // Chunk run.
    let (cs0, cs1) = (host(&state0)?, host(&vec![0f32; VH * HS * HS])?);
    let cout = ctx.create_buffer((NB * VH * HS * 4) as u64)?;
    let chunk_steps = vec![lev_vulkan::Step {
        kernel: &kc,
        bindings: vec![(&qkb, 0, qkb.size()), (&cvb, 0, cvb.size()), (&alb, 0, alb.size()), (&beb, 0, beb.size()), (&dtb, 0, dtb.size()), (&ab, 0, ab.size()), (&cs0, 0, cs0.size()), (&cs1, 0, cs1.size()), (&zb, 0, zb.size()), (&nwb, 0, nwb.size()), (&cout, 0, cout.size()), (&ctrl, 0, ctrl.size())],
        push: Vec::new(),
        groups: VH as u32,
    }];
    ctx.build_program(&chunk_steps, false)?.run()?;
    // Sequential run: the single-column kernel on each column in turn.
    let (ss0, ss1) = (host(&state0)?, host(&vec![0f32; VH * HS * HS])?);
    let sout = ctx.create_buffer((NB * VH * HS * 4) as u64)?;
    let mut keep = Vec::new();
    for c in 0..NB {
        let al = host(&alpha[c * VH..(c + 1) * VH])?;
        let be = host(&beta[c * VH..(c + 1) * VH])?;
        keep.push((al, be));
    }
    for c in 0..NB {
        let (al, be) = &keep[c];
        let steps = vec![lev_vulkan::Step {
            kernel: &k1,
            bindings: vec![
                (&qkb, (c * 2 * KH * HS * 4) as u64, (2 * KH * HS * 4) as u64),
                (&cvb, (c * CD * 4) as u64, (CD * 4) as u64),
                (al, 0, al.size()),
                (be, 0, be.size()),
                (&dtb, 0, dtb.size()),
                (&ab, 0, ab.size()),
                (&ss0, 0, ss0.size()),
                (&ss1, 0, ss1.size()),
                (&zb, (c * VH * HS * 4) as u64, (VH * HS * 4) as u64),
                (&nwb, 0, nwb.size()),
                (&sout, (c * VH * HS * 4) as u64, (VH * HS * 4) as u64),
                (&ctrl, 0, ctrl.size()),
            ],
            push: Vec::new(),
            groups: VH as u32,
        }];
        ctx.build_program(&steps, false)?.run()?;
    }
    let (co, so) = (cout.read_f32(0, NB * VH * HS), sout.read_f32(0, NB * VH * HS));
    let (cst, sst) = (cs0.read_f32(0, VH * HS * HS), ss0.read_f32(0, VH * HS * HS));
    let so64: Vec<f64> = so.iter().map(|&v| v as f64).collect();
    let sst64: Vec<f64> = sst.iter().map(|&v| v as f64).collect();
    let e_out = max_rel_err(&co, &so64).unwrap_or(f64::INFINITY);
    let e_state = max_rel_err(&cst, &sst64).unwrap_or(f64::INFINITY);
    println!("chunk vs sequential: output max_rel_err {e_out:.2e}, final state max_rel_err {e_state:.2e}");

    // Two-column kernel (decode verification pass) against the same sequential reference
    // for the first two columns: outputs of both columns and the state after column 1.
    let k2 = mk(&ops::ssm_delta(KH, VH, HS, 1e-6, CD, 2))?;
    let (ts0, ts1) = (host(&state0)?, host(&vec![0f32; VH * HS * HS])?);
    let tout = ctx.create_buffer((2 * VH * HS * 4) as u64)?;
    let two_steps = vec![lev_vulkan::Step {
        kernel: &k2,
        bindings: vec![(&qkb, 0, (2 * 2 * KH * HS * 4) as u64), (&cvb, 0, (2 * CD * 4) as u64), (&alb, 0, (2 * VH * 4) as u64), (&beb, 0, (2 * VH * 4) as u64), (&dtb, 0, dtb.size()), (&ab, 0, ab.size()), (&ts0, 0, ts0.size()), (&ts1, 0, ts1.size()), (&zb, 0, (2 * VH * HS * 4) as u64), (&nwb, 0, nwb.size()), (&tout, 0, tout.size()), (&ctrl, 0, ctrl.size())],
        push: Vec::new(),
        groups: VH as u32,
    }];
    ctx.build_program(&two_steps, false)?.run()?;
    // Sequential state after two columns: rerun two single-column steps from the initial state.
    let (rs0, rs1) = (host(&state0)?, host(&vec![0f32; VH * HS * HS])?);
    let rout = ctx.create_buffer((2 * VH * HS * 4) as u64)?;
    for c in 0..2 {
        let (al, be) = &keep[c];
        let steps = vec![lev_vulkan::Step {
            kernel: &k1,
            bindings: vec![
                (&qkb, (c * 2 * KH * HS * 4) as u64, (2 * KH * HS * 4) as u64),
                (&cvb, (c * CD * 4) as u64, (CD * 4) as u64),
                (al, 0, al.size()),
                (be, 0, be.size()),
                (&dtb, 0, dtb.size()),
                (&ab, 0, ab.size()),
                (&rs0, 0, rs0.size()),
                (&rs1, 0, rs1.size()),
                (&zb, (c * VH * HS * 4) as u64, (VH * HS * 4) as u64),
                (&nwb, 0, nwb.size()),
                (&rout, (c * VH * HS * 4) as u64, (VH * HS * 4) as u64),
                (&ctrl, 0, ctrl.size()),
            ],
            push: Vec::new(),
            groups: VH as u32,
        }];
        ctx.build_program(&steps, false)?.run()?;
    }
    let (t_out, r_out) = (tout.read_f32(0, 2 * VH * HS), rout.read_f32(0, 2 * VH * HS));
    let r_out64: Vec<f64> = r_out.iter().map(|&v| v as f64).collect();
    let e2_out = max_rel_err(&t_out, &r_out64).unwrap_or(f64::INFINITY);
    let e2_c0 = max_rel_err(&t_out[..VH * HS], &r_out64[..VH * HS]).unwrap_or(f64::INFINITY);
    let (t_s1, r_s0) = (ts1.read_f32(0, VH * HS * HS), rs0.read_f32(0, VH * HS * HS));
    let r_s064: Vec<f64> = r_s0.iter().map(|&v| v as f64).collect();
    let e2_state = max_rel_err(&t_s1, &r_s064).unwrap_or(f64::INFINITY);
    println!("2-column vs sequential: outputs {e2_out:.2e} (column 0 alone {e2_c0:.2e}), state after column 1 {e2_state:.2e}");
    // Three-column kernel: outputs of all columns and the state after each of them.
    let k3 = {
        let (spv, _) = compile_glsl_cached(&ops::ssm_delta(KH, VH, HS, 1e-6, CD, 3), &cache_dir)?;
        ctx.create_kernel(&spv, 13, 0)?
    };
    let (us0, us1, us2) = (host(&state0)?, host(&vec![0f32; VH * HS * HS])?, host(&vec![0f32; VH * HS * HS])?);
    let uout = ctx.create_buffer((3 * VH * HS * 4) as u64)?;
    let three_steps = vec![lev_vulkan::Step {
        kernel: &k3,
        bindings: vec![(&qkb, 0, (3 * 2 * KH * HS * 4) as u64), (&cvb, 0, (3 * CD * 4) as u64), (&alb, 0, (3 * VH * 4) as u64), (&beb, 0, (3 * VH * 4) as u64), (&dtb, 0, dtb.size()), (&ab, 0, ab.size()), (&us0, 0, us0.size()), (&us1, 0, us1.size()), (&zb, 0, (3 * VH * HS * 4) as u64), (&nwb, 0, nwb.size()), (&uout, 0, uout.size()), (&ctrl, 0, ctrl.size()), (&us2, 0, us2.size())],
        push: Vec::new(),
        groups: VH as u32,
    }];
    ctx.build_program(&three_steps, false)?.run()?;
    let (qs0, qout) = (host(&state0)?, ctx.create_buffer((3 * VH * HS * 4) as u64)?);
    let qs1 = host(&vec![0f32; VH * HS * HS])?;
    let mut e3_state = [0f64; 3];
    let mut snaps: Vec<Vec<f32>> = Vec::new();
    for c in 0..3 {
        let (al, be) = &keep[c];
        let steps = vec![lev_vulkan::Step {
            kernel: &k1,
            bindings: vec![
                (&qkb, (c * 2 * KH * HS * 4) as u64, (2 * KH * HS * 4) as u64),
                (&cvb, (c * CD * 4) as u64, (CD * 4) as u64),
                (al, 0, al.size()),
                (be, 0, be.size()),
                (&dtb, 0, dtb.size()),
                (&ab, 0, ab.size()),
                (&qs0, 0, qs0.size()),
                (&qs1, 0, qs1.size()),
                (&zb, (c * VH * HS * 4) as u64, (VH * HS * 4) as u64),
                (&nwb, 0, nwb.size()),
                (&qout, (c * VH * HS * 4) as u64, (VH * HS * 4) as u64),
                (&ctrl, 0, ctrl.size()),
            ],
            push: Vec::new(),
            groups: VH as u32,
        }];
        ctx.build_program(&steps, false)?.run()?;
        snaps.push(qs0.read_f32(0, VH * HS * HS));
    }
    let (t3_out, r3_out) = (uout.read_f32(0, 3 * VH * HS), qout.read_f32(0, 3 * VH * HS));
    let r3: Vec<f64> = r3_out.iter().map(|&v| v as f64).collect();
    let e3_out = max_rel_err(&t3_out, &r3).unwrap_or(f64::INFINITY);
    for (c, tb) in [&us0, &us1, &us2].iter().enumerate() {
        let want: Vec<f64> = snaps[c].iter().map(|&v| v as f64).collect();
        e3_state[c] = max_rel_err(&tb.read_f32(0, VH * HS * HS), &want).unwrap_or(f64::INFINITY);
    }
    println!("3-column vs sequential: outputs {e3_out:.2e}, state after column 0/1/2 {:.2e} / {:.2e} / {:.2e}", e3_state[0], e3_state[1], e3_state[2]);
    Ok(e_out < 1e-4 && e_state < 1e-4 && e2_out < 1e-4 && e2_state < 1e-4 && e3_out < 1e-4 && e3_state.iter().all(|&e| e < 1e-4))
}

/// Repeats prefill + plain decode + speculative decode on one resident model
/// for `seconds`, printing `SOAK,<unix ms>,<prefill tok/s>,<plain tok/s>,
/// <spec tok/s>,<draft acceptance>` per iteration, for time-series benchmarks.
fn gpu_soak(path: &str, seconds: u64, ids_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = parse_ids(std::fs::read_to_string(ids_file)?.trim())?;
    let mut gpu = lev_model::gpu::GpuModel::open(path, 1024, false)?;
    let end = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    let now_ms = || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
    let decode_rate = |st: &lev_model::gpu::SpecStats| (st.tokens.saturating_sub(1)) as f64 / (st.seconds - st.prefill_seconds);
    while std::time::Instant::now() < end {
        gpu.reset();
        let (_, ps) = gpu.generate(&prompt, 64, false)?;
        gpu.reset();
        let (_, ss) = gpu.generate(&prompt, 64, true)?;
        println!(
            "SOAK,{},{:.2},{:.2},{:.2},{:.3}",
            now_ms(),
            prompt.len() as f64 / ps.prefill_seconds,
            decode_rate(&ps),
            decode_rate(&ss),
            ss.accepted as f64 / ss.steps.max(1) as f64
        );
    }
    Ok(())
}

/// Intermediate buffers of the last single-token pass against column 1 of a
/// two-token pass (use with LEV_MAX_LAYERS to stop inside a layer range).
fn gpu_dbg(path: &str, ids: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = parse_ids(ids)?;
    let mut gpu = lev_model::gpu::GpuModel::open(path, 1024, false)?;
    let (head, pair) = tokens.split_at(tokens.len() - 2);
    for &t in head {
        gpu.forward(t)?;
    }
    gpu.forward(pair[0])?;
    gpu.forward(pair[1])?;
    let seq = gpu.debug_buffers()?;
    gpu.reset();
    for &t in head {
        gpu.forward(t)?;
    }
    gpu.forward_pair(pair[0], pair[1])?;
    let two = gpu.debug_buffers()?;
    println!("{:<10} {:>10} {:>12}", "buffer", "col size", "max_rel_err (pair col 1 vs sequential)");
    for ((name, s), (_, t)) in seq.iter().zip(&two) {
        // The pair pass stores two columns back to back; the sequential pass used column 0.
        let n = s.len() / 8; // buffers are sized for 8 or more columns; use the per-column stride of the pair pass
        let _ = n;
        let per = t.len().min(s.len());
        let col = per / lev_model::gpu::DEBUG_COLS;
        if col == 0 {
            continue;
        }
        let a: Vec<f64> = s[..col].iter().map(|&v| v as f64).collect();
        let b = &t[col..2 * col];
        let e = max_rel_err(b, &a).unwrap_or(f64::INFINITY);
        let mx = a.iter().fold(0f64, |m, v| m.max(v.abs()));
        println!("{name:<10} {col:>10} {e:>12.2e}   (max |ref| {mx:.2e})");
    }
    Ok(())
}

/// Writes the residual stream after every layer of the last token (sequential
/// single-token passes) to a binary file of f32, for `trace-diff`.
fn gpu_trace(path: &str, ids: &str, out: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = parse_ids(ids)?;
    let mut gpu = lev_model::gpu::GpuModel::open(path, 1024, true)?;
    for &t in &tokens {
        gpu.forward(t)?;
    }
    let layers = gpu.layer_outputs().ok_or("no trace")?;
    let bytes: Vec<u8> = layers.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, bytes)?;
    Ok(())
}

fn parse_ids(ids: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    Ok(ids.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?)
}

/// Runs the CPU reference and the GPU program on the same tokens and compares
/// the residual stream after every layer and the logits of the last token.
fn gpu_check(path: &str, ids: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let tokens = parse_ids(ids)?;
    let mut cpu = lev_model::Model::open(path)?;
    cpu.q8_activations = true;
    let mut gpu = lev_model::gpu::GpuModel::open(path, 512, true)?;
    println!("device {}, {} dispatches per token", gpu.device_name(), gpu.dispatch_count());
    let (mut cpu_logits, mut gpu_logits) = (Vec::new(), Vec::new());
    let mut worst_all = 0f64;
    for (i, &t) in tokens.iter().enumerate() {
        let mut cpu_trace = lev_model::Trace::default();
        cpu_logits = cpu.forward(t, Some(&mut cpu_trace))?;
        gpu_logits = gpu.forward(t)?;
        let gt = gpu.layer_outputs().ok_or("no trace")?;
        let mut worst = 0f64;
        for (l, (c, g)) in cpu_trace.layer_out.iter().zip(&gt).enumerate() {
            let c64: Vec<f64> = c.iter().map(|&v| v as f64).collect();
            let e = max_rel_err(g, &c64).unwrap_or(f64::INFINITY);
            worst = worst.max(e);
            if i + 1 == tokens.len() {
                println!("layer {l:>2}: max_rel_err {e:.2e}");
            }
        }
        worst_all = worst_all.max(worst);
        println!("token {i} (id {t}): worst layer error {worst:.2e}, gpu {:.1} ms", gpu.last_gpu_us / 1000.0);
    }
    let c64: Vec<f64> = cpu_logits.iter().map(|&v| v as f64).collect();
    let lerr = max_rel_err(&gpu_logits, &c64).unwrap_or(f64::INFINITY);
    let top_cpu = lev_model::log_softmax_top(&cpu_logits, 5);
    let top_gpu = lev_model::log_softmax_top(&gpu_logits, 5);
    println!("logits max_rel_err {lerr:.2e}");
    for (c, g) in top_cpu.iter().zip(&top_gpu) {
        println!("cpu id {:>7} logprob {:>8.4} | gpu id {:>7} logprob {:>8.4}", c.0, c.2, g.0, g.2);
    }
    let same_top = top_cpu.iter().map(|x| x.0).eq(top_gpu.iter().map(|x| x.0));
    println!("top-5 order {} ; worst layer error over all tokens {worst_all:.2e}", if same_top { "identical" } else { "DIFFERS" });
    Ok(same_top && lerr < 1e-2)
}

/// Greedy decode: prompt tokens first, then `n` generated tokens; reports the
/// decode rate (wall clock, readback and argmax included) and GPU time.
fn gpu_bench(path: &str, ids: &str, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = parse_ids(ids)?;
    let mut gpu = lev_model::gpu::GpuModel::open(path, std::env::var("LEV_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(1024), false)?;
    println!("device {}, {} dispatches per token", gpu.device_name(), gpu.dispatch_count());
    let mut next = 0usize;
    let t0 = std::time::Instant::now();
    for &t in &prompt {
        let l = gpu.forward(t)?;
        next = lev_model::argmax(&l);
    }
    let prefill = t0.elapsed().as_secs_f64();
    let mut out = Vec::new();
    let mut gpu_us = 0.0;
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        out.push(next);
        let l = gpu.forward(next)?;
        gpu_us += gpu.last_gpu_us;
        next = lev_model::argmax(&l);
    }
    let dt = t1.elapsed().as_secs_f64();
    println!("prompt {} tokens: {:.2} s ({:.1} tok/s, token by token)", prompt.len(), prefill, prompt.len() as f64 / prefill);
    println!("generated ids: {out:?}");
    println!("decode: {n} tokens in {dt:.2} s = {:.2} tok/s wall; GPU program {:.2} ms/token", n as f64 / dt, gpu_us / n as f64 / 1000.0);
    Ok(())
}
/// Forge loop for the int8-activation Q4_K matvec: quantize the input on the
/// GPU, run the integer dot-product kernel, gate against an oracle that
/// quantizes the input the same way, then time quantize + matvec.
fn vk_q8(path: &str, name: &str, max_rows: usize, nb: u32) -> Result<bool, Box<dyn std::error::Error>> {
    use lev_cpu::{matvec_q4k_q8_f64, quantize_q8};
    use lev_forge::q8;
    let g = GgufFile::open(path)?;
    let t = g.tensor(name).ok_or("tensor not found")?;
    let is_q6 = t.ty == GgmlType::Q6K;
    if !(t.ty == GgmlType::Q4K || is_q6) || t.dims.len() != 2 {
        return Err(format!("{name}: expected a 2-D Q4_K or Q6_K tensor").into());
    }
    let block_bytes = if is_q6 { lev_cpu::Q6K_BLOCK_BYTES } else { Q4K_BLOCK_BYTES };
    let (cols, total_rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let rows = max_rows.min(total_rows);
    let bpr = cols / QK_K;
    let mut file = File::open(path)?;
    let weights = g.read_tensor_bytes(&mut file, t, 0, (rows * bpr * block_bytes) as u64)?;
    let mut rng = SplitMix64(0x4C45_5649);
    let nbu = nb as usize;
    let x: Vec<f32> = (0..cols * nbu).map(|_| rng.next_signed_unit()).collect();
    let oracles: Vec<Vec<f64>> = (0..nbu).map(|c| (if is_q6 { lev_cpu::matvec_q6k_q8_f64 } else { matvec_q4k_q8_f64 })(&weights, rows, cols, &quantize_q8(&x[c * cols..(c + 1) * cols]))).collect::<Result<_, _>>()?;
    let float_ref: Vec<Vec<f64>> = (0..nbu).map(|c| (if is_q6 { lev_cpu::matvec_q6k_f64 } else { matvec_q4k_f64 })(&weights, rows, cols, &x[c * cols..(c + 1) * cols])).collect::<Result<_, _>>()?;

    let ctx = VulkanContext::open(&DeviceSelector::Auto)?;
    let info = ctx.info().clone();
    if !info.int_dot {
        return Err("device lacks VK_KHR_shader_integer_dot_product".into());
    }
    let cache_dir = std::env::var_os("LEV_CACHE_DIR")
        .map(|p| std::path::PathBuf::from(p).join("spv"))
        .unwrap_or_else(|| std::env::temp_dir().join("lev-forge").join("spv"));
    let wbuf = ctx.create_device_buffer(weights.len() as u64 + 16)?;
    ctx.upload(&wbuf, 0, &weights)?;
    let xbuf = ctx.create_buffer((x.len() * 4) as u64)?;
    xbuf.write_f32(0, &x);
    let xq = ctx.create_device_buffer((x.len()) as u64)?;
    let xs = ctx.create_device_buffer((x.len() / 32 * 8) as u64)?;
    let ybuf = ctx.create_buffer((rows * nbu * 4) as u64)?;
    let (qspv, _) = compile_glsl_cached(&q8::emit_quantize(), &cache_dir)?;
    let qk = ctx.create_kernel(&qspv, 3, 0)?;
    println!("tensor {name}: {} [{cols} x {total_rows}], {rows} rows, nb={nb}, {:.1} MB of weights", t.ty.name(), weights.len() as f64 / 1e6);
    println!("{:<14} {:<6} {:>12} {:>12} {:>10} {:>9} {}", "schedule", "cache", "vs_q8_oracle", "vs_float", "median_us", "GB/s", "gate");
    let candidates = q8::derive(&info.caps, nb);
    let mut best: Option<(usize, f64)> = None;
    let mut all_pass = true;
    for (i, s) in candidates.iter().enumerate() {
        let (spv, hit) = compile_glsl_cached(&(if is_q6 { q8::emit_q6k_q8(s)? } else { q8::emit_q4k_q8(s)? }), &cache_dir)?;
        let mk = ctx.create_kernel(&spv, 4, 8)?;
        let push = [(rows as u32).to_le_bytes(), (bpr as u32).to_le_bytes()].concat();
        let steps = vec![
            lev_vulkan::Step { kernel: &qk, bindings: vec![(&xbuf, 0, xbuf.size()), (&xq, 0, xq.size()), (&xs, 0, xs.size())], push: Vec::new(), groups: (x.len() / 256) as u32 },
            lev_vulkan::Step { kernel: &mk, bindings: vec![(&wbuf, 0, wbuf.size()), (&xq, 0, xq.size()), (&ybuf, 0, ybuf.size()), (&xs, 0, xs.size())], push, groups: q8::group_count(s, rows as u32) },
        ];
        let prog = ctx.build_program(&steps, false)?;
        let mut times = Vec::new();
        for _ in 0..3 {
            prog.run()?;
        }
        for _ in 0..11 {
            times.push(prog.run()?);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[times.len() / 2];
        let y = ybuf.read_f32(0, rows * nbu);
        let mut e_q = 0f64;
        let mut e_f = 0f64;
        for c in 0..nbu {
            let col = &y[c * rows..(c + 1) * rows];
            e_q = e_q.max(max_rel_err(col, &oracles[c]).unwrap_or(f64::INFINITY));
            e_f = e_f.max(max_rel_err(col, &float_ref[c]).unwrap_or(f64::INFINITY));
        }
        let pass = e_q < GATE;
        all_pass &= pass;
        println!(
            "T={:<2} R={:<2} SG={} {:<6} {:>12.2e} {:>12.2e} {:>10.1} {:>9.1} {}",
            s.t, s.r, s.sg, if hit { "hit" } else { "miss" }, e_q, e_f, med, weights.len() as f64 / (med * 1e-6) / 1e9, if pass { "PASS" } else { "FAIL" }
        );
        if pass && best.is_none_or(|(_, b)| med < b) {
            best = Some((i, med));
        }
    }
    match best {
        Some((i, v)) => println!("selected: T={} R={} SG={} ({:.1} us)", candidates[i].t, candidates[i].r, candidates[i].sg, v),
        None => println!("selected: none"),
    }
    Ok(all_pass)
}
/// Streaming-read probe: one workgroup of 64 per row reads the row's bytes as
/// coalesced uvec4 and writes a checksum. It measures the read bandwidth this
/// device actually delivers, the ceiling a matvec kernel can be compared to.
fn vk_bw(path: &str, rows: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Probe variants: load width in bytes per lane (4, 8, 16) and lanes per row.
    fn probe(width: u32, lanes: u32) -> String {
        let (ty, comps) = match width {
            4 => ("uint", "v"),
            8 => ("uvec2", "v.x ^ v.y"),
            _ => ("uvec4", "v.x ^ v.y ^ v.z ^ v.w"),
        };
        format!(
            "#version 450
layout(local_size_x = {lanes}, local_size_y = 1, local_size_z = 1) in;
layout(std430, set = 0, binding = 0) readonly buffer Weights {{ {ty} w[]; }};
layout(std430, set = 0, binding = 1) readonly buffer Input {{ float x[]; }};
layout(std430, set = 0, binding = 2) writeonly buffer Output {{ float y[]; }};
layout(push_constant) uniform Params {{ uint rows; uint blocks_per_row; }} p;
shared uint part[{lanes}];
void main() {{
    uint row = gl_WorkGroupID.x;
    uint t = gl_LocalInvocationID.x;
    uint n = p.blocks_per_row * 144u / {width}u;
    uint acc = 0u;
    for (uint i = t; i < n; i += {lanes}u) {{
        {ty} v = w[row * n + i];
        acc += {comps};
    }}
    part[t] = acc;
    barrier();
    for (uint k = {lanes}u / 2u; k > 0u; k >>= 1u) {{
        if (t < k) {{ part[t] += part[t + k]; }}
        barrier();
    }}
    if (t == 0u) {{ y[row] = float(part[0] & 255u); }}
}}
"
        )
    }
    let g = GgufFile::open(path)?;
    let t = g.tensor("blk.0.ffn_gate.weight").ok_or("tensor not found")?;
    let (cols, total_rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let rows = rows.min(total_rows);
    let bpr = cols / QK_K;
    let mut file = File::open(path)?;
    let weights = g.read_tensor_bytes(&mut file, t, 0, (rows * bpr * Q4K_BLOCK_BYTES) as u64)?;
    let ctx = VulkanContext::open(&DeviceSelector::Auto)?;
    let cache_dir = std::env::var_os("LEV_CACHE_DIR")
        .map(|p| std::path::PathBuf::from(p).join("spv"))
        .unwrap_or_else(|| std::env::temp_dir().join("lev-forge").join("spv"));
    // r=1, rr=1: one workgroup per row.
    let s = lev_forge::schedule::Q4kSchedule { t: 64, r: 1, xv: 1, m: 0, rr: 1, sg: 0, u: 1, nb: 1 };
    let x = vec![0f32; cols];
    println!("streaming read of {:.1} MB, one workgroup per row (median of 21 runs)", weights.len() as f64 / 1e6);
    for width in [4u32, 8, 16] {
        for lanes in [16u32, 32, 64, 128] {
            let (spirv, _) = compile_glsl_cached(&probe(width, lanes), &cache_dir)?;
            let run = ctx.run_q4k_matvec(&spirv, &s, &weights, rows as u32, cols as u32, &x, 3, 21)?;
            let med = run.median_us();
            println!("load {:>2} B/lane, {:>3} lanes/row: {:>7.1} us = {:>6.1} GB/s", width, lanes, med, weights.len() as f64 / (med * 1e-6) / 1e9);
        }
    }
    Ok(())
}
/// Per-candidate Forge loop on the GPU: emit, compile (cached), run, gate
/// against the FP64 oracle, then time. A candidate is only eligible for
/// selection if it passes the gate; a non-baseline candidate must beat the
/// baseline median by more than 2%.
fn vk_q4k(path: &str, name: &str, max_rows: usize, device: Option<&str>) -> Result<bool, Box<dyn std::error::Error>> {
    let g = GgufFile::open(path)?;
    let t = g.tensor(name).ok_or("tensor not found")?;
    if t.ty != GgmlType::Q4K || t.dims.len() != 2 {
        return Err(format!("{name}: expected a 2-D Q4_K tensor, got {} {:?}", t.ty.name(), t.dims).into());
    }
    let (cols, total_rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let rows = max_rows.min(total_rows);
    let blocks_per_row = cols / QK_K;
    let mut file = File::open(path)?;
    let weights = g.read_tensor_bytes(&mut file, t, 0, (rows * blocks_per_row * Q4K_BLOCK_BYTES) as u64)?;

    let mut rng = SplitMix64(0x4C45_5649);
    let x: Vec<f32> = (0..cols).map(|_| rng.next_signed_unit()).collect();
    let cpu = matvec_q4k_f32(&weights, rows, cols, &x)?;
    let oracle = matvec_q4k_f64(&weights, rows, cols, &x)?;
    let cpu64: Vec<f64> = cpu.iter().map(|&v| v as f64).collect();

    let selector = match device {
        None => DeviceSelector::Auto,
        Some(s) => s.parse::<usize>().map(DeviceSelector::Index).unwrap_or_else(|_| DeviceSelector::NameContains(s.to_string())),
    };
    let ctx = VulkanContext::open(&selector)?;
    let info = ctx.info().clone();
    println!(
        "device [{}] {} ({}), subgroup={}, max_invocations={}",
        info.index, info.name, info.device_type, info.caps.subgroup_size, info.caps.max_workgroup_invocations
    );
    let cache_dir = std::env::var_os("LEV_CACHE_DIR")
        .map(|p| std::path::PathBuf::from(p).join("spv"))
        .unwrap_or_else(|| std::env::temp_dir().join("lev-forge").join("spv"));
    println!(
        "tensor {name}: Q4_K [{cols} x {total_rows}], {rows} rows, {:.1} MB of weights, gate < {GATE:e}",
        weights.len() as f64 / 1e6
    );

    let candidates = lev_forge::schedule::derive(blocks_per_row, &info.caps);
    println!("{:<10} {:<6} {:>12} {:>10} {:>10} {:>12} {}", "schedule", "cache", "max_rel_err", "vs_cpu", "median_us", "GB/s", "gate");
    let mut measured: Vec<Option<f64>> = Vec::new();
    let mut all_pass = true;
    for s in &candidates {
        let src = lev_forge::glsl::emit_q4k_matvec(s)?;
        let (spirv, hit) = compile_glsl_cached(&src, &cache_dir)?;
        let run = ctx.run_q4k_matvec(&spirv, s, &weights, rows as u32, cols as u32, &x, 3, 11)?;
        let err = max_rel_err(&run.y, &oracle);
        let vs_cpu = max_rel_err(&run.y, &cpu64);
        let pass = err.is_some_and(|e| e < GATE);
        all_pass &= pass;
        let med = run.median_us();
        let gbs = weights.len() as f64 / (med * 1e-6) / 1e9;
        println!(
            "T={:<3} R={:<2} V={} M={} RR={} SG={} U={} {:<6} {:>12} {:>10} {:>10.1} {:>12.1} {}",
            s.t,
            s.r,
            s.xv,
            s.m,
            s.rr,
            s.sg,
            s.u,
            if hit { "hit" } else { "miss" },
            err.map_or("rejected".to_string(), |e| format!("{e:.2e}")),
            vs_cpu.map_or("rejected".to_string(), |e| format!("{e:.1e}")),
            med,
            gbs,
            if pass { "PASS" } else { "FAIL" }
        );
        measured.push(pass.then_some(med));
    }

    let baseline = measured.first().copied().flatten();
    let best = measured
        .iter()
        .enumerate()
        .filter_map(|(i, m)| m.map(|v| (i, v)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    match (baseline, best) {
        (Some(base), Some((i, v))) if i != 0 && v < base * 0.98 => {
            println!("selected: T={} R={} V={} M={} RR={} SG={} U={} ({:.1}% faster than the baseline T=1 R=1 V=1)", candidates[i].t, candidates[i].r, candidates[i].xv, candidates[i].m, candidates[i].rr, candidates[i].sg, candidates[i].u, (1.0 - v / base) * 100.0);
        }
        (Some(_), _) => println!("selected: baseline T=1 R=1 (no candidate beat it by more than 2%)"),
        (None, _) => println!("selected: none (baseline did not pass the gate)"),
    }
    Ok(all_pass)
}
