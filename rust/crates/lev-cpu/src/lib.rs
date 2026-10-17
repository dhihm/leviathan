//! CPU reference kernels and independent FP64 oracles.
//!
//! The f32 matvec fixes one deterministic summation order (ascending blocks,
//! ascending elements) and is the ground truth GPU candidates are compared
//! against; the f64 oracle recomputes everything in double precision from the
//! raw block bytes. Both read the same documented block layout, so agreement
//! checks arithmetic, not the layout itself (the layout is checked against
//! real model logits at a later milestone).

pub const QK_K: usize = 256;
pub const Q4K_BLOCK_BYTES: usize = 144;

/// IEEE half to double, exact for every finite value.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f64;
    match exp {
        0 => sign * mant * 2f64.powi(-24),
        31 if mant == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        e => sign * (1.0 + mant / 1024.0) * 2f64.powi(e - 15),
    }
}

/// f32 to IEEE half bits, round to nearest even; overflow gives infinity.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = if rem > mid || (rem == mid && (half & 1) == 1) { 1 } else { 0 };
        return sign | (half + round);
    }
    let half = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    let round = if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) { 1 } else { 0 };
    sign | (half + round) as u16
}

/// Exact: every half value is representable in f32.
pub fn f16_to_f32(h: u16) -> f32 {
    f16_to_f64(h) as f32
}

/// Unpacks the 6-bit scale and min of sub-block `j` (0..8) from the 12 packed
/// scale bytes of a Q4_K block.
pub fn unpack_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    debug_assert!(scales.len() >= 12 && j < 8);
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ShapeError {
    ColsNotBlockMultiple(usize),
    WeightBytes { expected: usize, got: usize },
    InputLen { expected: usize, got: usize },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColsNotBlockMultiple(c) => write!(f, "cols {c} is not a multiple of {QK_K}"),
            Self::WeightBytes { expected, got } => write!(f, "weight bytes: expected {expected}, got {got}"),
            Self::InputLen { expected, got } => write!(f, "input length: expected {expected}, got {got}"),
        }
    }
}

impl std::error::Error for ShapeError {}

fn check_shape(weights: &[u8], rows: usize, cols: usize, x_len: usize) -> Result<usize, ShapeError> {
    if cols == 0 || cols % QK_K != 0 {
        return Err(ShapeError::ColsNotBlockMultiple(cols));
    }
    let row_bytes = cols / QK_K * Q4K_BLOCK_BYTES;
    let expected = rows.checked_mul(row_bytes).ok_or(ShapeError::WeightBytes { expected: usize::MAX, got: weights.len() })?;
    if weights.len() != expected {
        return Err(ShapeError::WeightBytes { expected, got: weights.len() });
    }
    if x_len != cols {
        return Err(ShapeError::InputLen { expected: cols, got: x_len });
    }
    Ok(row_bytes)
}

/// Dequantizes one 144-byte Q4_K block (256 values) in f32.
pub fn dequant_block_f32(block: &[u8], out: &mut [f32; QK_K]) {
    debug_assert_eq!(block.len(), Q4K_BLOCK_BYTES);
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..Q4K_BLOCK_BYTES];
    let mut is = 0;
    for chunk in 0..4 {
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let (sc1, m1) = unpack_scale_min(is, scales);
        let (sc2, m2) = unpack_scale_min(is + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
        let o = chunk * 64;
        for l in 0..32 {
            out[o + l] = d1 * (q[l] & 0x0F) as f32 - mm1;
            out[o + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
        }
        is += 2;
    }
}

/// Dequantizes one Q4_K block in f64 (oracle).
pub fn dequant_block_f64(block: &[u8], out: &mut [f64; QK_K]) {
    debug_assert_eq!(block.len(), Q4K_BLOCK_BYTES);
    let d = f16_to_f64(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f64(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..Q4K_BLOCK_BYTES];
    let mut is = 0;
    for chunk in 0..4 {
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let (sc1, m1) = unpack_scale_min(is, scales);
        let (sc2, m2) = unpack_scale_min(is + 1, scales);
        let (d1, mm1) = (d * sc1 as f64, dmin * m1 as f64);
        let (d2, mm2) = (d * sc2 as f64, dmin * m2 as f64);
        let o = chunk * 64;
        for l in 0..32 {
            out[o + l] = d1 * (q[l] & 0x0F) as f64 - mm1;
            out[o + 32 + l] = d2 * (q[l] >> 4) as f64 - mm2;
        }
        is += 2;
    }
}

/// `y[row] = sum_col W[row, col] * x[col]` with W in Q4_K blocks (row-major),
/// f32 arithmetic in ascending block and element order.
pub fn matvec_q4k_f32(weights: &[u8], rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>, ShapeError> {
    let row_bytes = check_shape(weights, rows, cols, x.len())?;
    let mut y = Vec::with_capacity(rows);
    let mut tmp = [0f32; QK_K];
    for r in 0..rows {
        let row = &weights[r * row_bytes..(r + 1) * row_bytes];
        let mut sum = 0f32;
        for (b, block) in row.chunks_exact(Q4K_BLOCK_BYTES).enumerate() {
            dequant_block_f32(block, &mut tmp);
            let xs = &x[b * QK_K..(b + 1) * QK_K];
            for i in 0..QK_K {
                sum += tmp[i] * xs[i];
            }
        }
        y.push(sum);
    }
    Ok(y)
}

/// Independent double-precision oracle; the input is widened exactly.
pub fn matvec_q4k_f64(weights: &[u8], rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f64>, ShapeError> {
    let row_bytes = check_shape(weights, rows, cols, x.len())?;
    let mut y = Vec::with_capacity(rows);
    let mut tmp = [0f64; QK_K];
    for r in 0..rows {
        let row = &weights[r * row_bytes..(r + 1) * row_bytes];
        let mut sum = 0f64;
        for (b, block) in row.chunks_exact(Q4K_BLOCK_BYTES).enumerate() {
            dequant_block_f64(block, &mut tmp);
            let xs = &x[b * QK_K..(b + 1) * QK_K];
            for i in 0..QK_K {
                sum += tmp[i] * xs[i] as f64;
            }
        }
        y.push(sum);
    }
    Ok(y)
}

pub const Q6K_BLOCK_BYTES: usize = 210;

/// Unpacks one 210-byte Q6_K block into signed 6-bit quants (-32..=31) in
/// element order. Layout: 128 bytes of low nibbles, 64 bytes of high 2-bit
/// pairs, 16 signed scale bytes, one f16 `d` at the end. Element `e` uses
/// scale `e / 16`.
pub fn q6k_unpack(block: &[u8], q: &mut [i8; QK_K]) {
    debug_assert_eq!(block.len(), Q6K_BLOCK_BYTES);
    let (ql, qh) = (&block[0..128], &block[128..192]);
    for half in 0..2 {
        let (ql, qh) = (&ql[half * 64..], &qh[half * 32..]);
        for l in 0..32 {
            let h = qh[l];
            q[half * 128 + l] = ((ql[l] & 0x0F) | ((h & 3) << 4)) as i8 - 32;
            q[half * 128 + l + 32] = ((ql[l + 32] & 0x0F) | (((h >> 2) & 3) << 4)) as i8 - 32;
            q[half * 128 + l + 64] = ((ql[l] >> 4) | (((h >> 4) & 3) << 4)) as i8 - 32;
            q[half * 128 + l + 96] = ((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i8 - 32;
        }
    }
}

fn q6k_scales_and_d(block: &[u8]) -> ([i8; 16], f64) {
    let mut sc = [0i8; 16];
    for (i, s) in sc.iter_mut().enumerate() {
        *s = block[192 + i] as i8;
    }
    (sc, f16_to_f64(u16::from_le_bytes([block[208], block[209]])))
}

pub fn dequant_q6k_block_f32(block: &[u8], out: &mut [f32; QK_K]) {
    let mut q = [0i8; QK_K];
    q6k_unpack(block, &mut q);
    let (sc, d) = q6k_scales_and_d(block);
    let d = d as f32;
    for e in 0..QK_K {
        out[e] = d * sc[e / 16] as f32 * q[e] as f32;
    }
}

pub fn dequant_q6k_block_f64(block: &[u8], out: &mut [f64; QK_K]) {
    let mut q = [0i8; QK_K];
    q6k_unpack(block, &mut q);
    let (sc, d) = q6k_scales_and_d(block);
    for e in 0..QK_K {
        out[e] = d * sc[e / 16] as f64 * q[e] as f64;
    }
}

fn check_shape_q6k(weights: &[u8], rows: usize, cols: usize, x_len: usize) -> Result<usize, ShapeError> {
    if cols == 0 || cols % QK_K != 0 {
        return Err(ShapeError::ColsNotBlockMultiple(cols));
    }
    let row_bytes = cols / QK_K * Q6K_BLOCK_BYTES;
    let expected = rows.checked_mul(row_bytes).ok_or(ShapeError::WeightBytes { expected: usize::MAX, got: weights.len() })?;
    if weights.len() != expected {
        return Err(ShapeError::WeightBytes { expected, got: weights.len() });
    }
    if x_len != cols {
        return Err(ShapeError::InputLen { expected: cols, got: x_len });
    }
    Ok(row_bytes)
}

/// Q6_K matvec in f32, ascending blocks and elements.
pub fn matvec_q6k_f32(weights: &[u8], rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>, ShapeError> {
    let row_bytes = check_shape_q6k(weights, rows, cols, x.len())?;
    let mut y = Vec::with_capacity(rows);
    let mut tmp = [0f32; QK_K];
    for r in 0..rows {
        let mut sum = 0f32;
        for (b, block) in weights[r * row_bytes..(r + 1) * row_bytes].chunks_exact(Q6K_BLOCK_BYTES).enumerate() {
            dequant_q6k_block_f32(block, &mut tmp);
            for i in 0..QK_K {
                sum += tmp[i] * x[b * QK_K + i];
            }
        }
        y.push(sum);
    }
    Ok(y)
}

/// Q6_K matvec oracle in f64.
pub fn matvec_q6k_f64(weights: &[u8], rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f64>, ShapeError> {
    let row_bytes = check_shape_q6k(weights, rows, cols, x.len())?;
    let mut y = Vec::with_capacity(rows);
    let mut tmp = [0f64; QK_K];
    for r in 0..rows {
        let mut sum = 0f64;
        for (b, block) in weights[r * row_bytes..(r + 1) * row_bytes].chunks_exact(Q6K_BLOCK_BYTES).enumerate() {
            dequant_q6k_block_f64(block, &mut tmp);
            for i in 0..QK_K {
                sum += tmp[i] * x[b * QK_K + i] as f64;
            }
        }
        y.push(sum);
    }
    Ok(y)
}
/// Activation vector quantized to int8 in groups of 32, the form the integer
/// dot-product matvec kernels consume. `d[g]` is the group scale and
/// `s[g] = d[g] * sum(q in group)`, the group's contribution to the min term.
pub struct Q8Activations {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    pub s: Vec<f32>,
}

/// Quantizes `x` (length a multiple of 32): `d = amax / 127`,
/// `q = round_ties_even(x * (127 / amax))`, the arithmetic the GPU kernel uses.
pub fn quantize_q8(x: &[f32]) -> Q8Activations {
    assert!(x.len() % 32 == 0);
    let mut out = Q8Activations { q: Vec::with_capacity(x.len()), d: Vec::new(), s: Vec::new() };
    for g in x.chunks_exact(32) {
        let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (d, id) = if amax > 0.0 { (amax / 127.0, 127.0 / amax) } else { (0.0, 0.0) };
        let mut sum = 0i32;
        for v in g {
            let q = (v * id).round_ties_even() as i32;
            sum += q;
            out.q.push(q as i8);
        }
        out.d.push(d);
        out.s.push(d * sum as f32);
    }
    out
}

/// Oracle for the int8-activation Q4_K matvec: exact (f64) product of the
/// dequantized weights and the dequantized quantized activations.
pub fn matvec_q4k_q8_f64(weights: &[u8], rows: usize, cols: usize, x: &Q8Activations) -> Result<Vec<f64>, ShapeError> {
    let xd: Vec<f32> = x.q.iter().enumerate().map(|(i, &q)| q as f32 * x.d[i / 32]).collect();
    // dequantized activations are exactly representable in f32 products only
    // approximately; widen the same f32 values the kernel effectively uses.
    matvec_q4k_f64(weights, rows, cols, &xd)
}

/// Oracle for the int8-activation Q6_K matvec.
pub fn matvec_q6k_q8_f64(weights: &[u8], rows: usize, cols: usize, x: &Q8Activations) -> Result<Vec<f64>, ShapeError> {
    let xd: Vec<f32> = x.q.iter().enumerate().map(|(i, &q)| q as f32 * x.d[i / 32]).collect();
    matvec_q6k_f64(weights, rows, cols, &xd)
}
/// Worst `|got - ref| / max(1, |ref|)`, the repository's mixed absolute and
/// relative contract. Rejects (None) length mismatch, empty input and any
/// non-finite element, so NaN cannot pass a numeric comparison.
pub fn max_rel_err(got: &[f32], reference: &[f64]) -> Option<f64> {
    if got.len() != reference.len() || got.is_empty() {
        return None;
    }
    let mut worst = 0f64;
    for (g, r) in got.iter().zip(reference) {
        if !g.is_finite() || !r.is_finite() {
            return None;
        }
        worst = worst.max((*g as f64 - r).abs() / r.abs().max(1.0));
    }
    Some(worst)
}

/// Deterministic SplitMix64 stream (same recipe as the Swift side's RNG.swift
/// purpose: byte-identical inputs across runs and machines).
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    pub fn next_signed_unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ggml encoder's packing of eight 6-bit scales and mins into 12 bytes.
    fn pack_scale_min(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
        let mut s = [0u8; 12];
        for j in 0..8 {
            let (ls, lm) = (sc[j], mn[j]);
            if j < 4 {
                s[j] = ls;
                s[j + 4] = lm;
            } else {
                s[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                s[j - 4] |= (ls >> 4) << 6;
                s[j] |= (lm >> 4) << 6;
            }
        }
        s
    }

    fn random_blocks(rng: &mut SplitMix64, n: usize) -> Vec<u8> {
        // Small finite half values for d and dmin.
        const HALVES: [u16; 6] = [0x2E66, 0x3000, 0x3266, 0x2800, 0x1C00, 0x3400];
        let mut out = Vec::with_capacity(n * Q4K_BLOCK_BYTES);
        for _ in 0..n {
            out.extend_from_slice(&HALVES[(rng.next_u64() % 6) as usize].to_le_bytes());
            out.extend_from_slice(&HALVES[(rng.next_u64() % 6) as usize].to_le_bytes());
            for _ in 0..(Q4K_BLOCK_BYTES - 4) {
                out.push(rng.next_u64() as u8);
            }
        }
        out
    }

    #[test]
    fn half_conversion_known_values() {
        assert_eq!(f16_to_f64(0x3C00), 1.0);
        assert_eq!(f16_to_f64(0xC000), -2.0);
        assert_eq!(f16_to_f64(0x0001), 2f64.powi(-24));
        assert_eq!(f16_to_f64(0x7BFF), 65504.0);
        assert_eq!(f16_to_f64(0x7C00), f64::INFINITY);
        assert!(f16_to_f64(0x7E00).is_nan());
        assert_eq!(f16_to_f64(0x0000), 0.0);
    }

    #[test]
    fn scale_min_pack_unpack_roundtrip() {
        let mut rng = SplitMix64(7);
        for _ in 0..2000 {
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..8 {
                sc[j] = (rng.next_u64() % 64) as u8;
                mn[j] = (rng.next_u64() % 64) as u8;
            }
            let packed = pack_scale_min(&sc, &mn);
            for j in 0..8 {
                assert_eq!(unpack_scale_min(j, &packed), (sc[j], mn[j]), "sub-block {j}");
            }
        }
    }

    #[test]
    fn hand_computed_block() {
        // d = 1.0, dmin = 0.5, every scale = 1, every min = 2,
        // qs bytes 0x21: low nibble 1, high nibble 2.
        let sc = [1u8; 8];
        let mn = [2u8; 8];
        let mut block = vec![0u8; Q4K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        block[4..16].copy_from_slice(&pack_scale_min(&sc, &mn));
        for b in &mut block[16..] {
            *b = 0x21;
        }
        let mut f32out = [0f32; QK_K];
        let mut f64out = [0f64; QK_K];
        dequant_block_f32(&block, &mut f32out);
        dequant_block_f64(&block, &mut f64out);
        for chunk in 0..4 {
            for l in 0..32 {
                // low: 1*1*1 - 0.5*2 = 0 ; high: 1*1*2 - 0.5*2 = 1
                assert_eq!(f64out[chunk * 64 + l], 0.0);
                assert_eq!(f64out[chunk * 64 + 32 + l], 1.0);
                assert_eq!(f32out[chunk * 64 + l], 0.0);
                assert_eq!(f32out[chunk * 64 + 32 + l], 1.0);
            }
        }
    }

    #[test]
    fn matvec_f32_matches_f64_oracle() {
        let mut rng = SplitMix64(42);
        let (rows, cols) = (16, 1024);
        let weights = random_blocks(&mut rng, rows * cols / QK_K);
        let x: Vec<f32> = (0..cols).map(|_| rng.next_signed_unit()).collect();
        let got = matvec_q4k_f32(&weights, rows, cols, &x).unwrap();
        let reference = matvec_q4k_f64(&weights, rows, cols, &x).unwrap();
        let err = max_rel_err(&got, &reference).unwrap();
        assert!(err < 1e-4, "max relative error {err}");
    }

    #[test]
    fn gate_rejects_non_finite_and_mismatch() {
        assert_eq!(max_rel_err(&[f32::NAN], &[0.0]), None);
        assert_eq!(max_rel_err(&[1.0], &[f64::INFINITY]), None);
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0]), None);
        assert_eq!(max_rel_err(&[], &[]), None);
        assert!(max_rel_err(&[1.0], &[1.0]).unwrap() == 0.0);
    }

    #[test]
    fn shape_errors() {
        let x = vec![0f32; 256];
        assert_eq!(matvec_q4k_f32(&[], 1, 100, &x[..100]).unwrap_err(), ShapeError::ColsNotBlockMultiple(100));
        assert!(matches!(matvec_q4k_f32(&[0u8; 10], 1, 256, &x), Err(ShapeError::WeightBytes { .. })));
        assert!(matches!(matvec_q4k_f32(&[0u8; 144], 1, 256, &x[..10]), Err(ShapeError::InputLen { .. })));
    }

    #[test]
    fn q6k_hand_computed_block() {
        // ql low nibbles = 5 everywhere, qh = 0 -> q = 5 - 32 = -27 for the
        // low-nibble elements; high-nibble elements (ql >> 4) = 5 too.
        let mut block = vec![0u8; Q6K_BLOCK_BYTES];
        for b in &mut block[0..128] {
            *b = 0x55;
        }
        for (i, b) in block[192..208].iter_mut().enumerate() {
            *b = (i as i8 - 8) as u8; // scales -8..=7
        }
        block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        let mut out = [0f64; QK_K];
        dequant_q6k_block_f64(&block, &mut out);
        for e in 0..QK_K {
            assert_eq!(out[e], (e / 16) as f64 * 0.0 + ((e / 16) as i32 - 8) as f64 * -27.0, "element {e}");
        }
        // High bits: qh = 0b11_10_01_00 for every byte gives per-group extra 0,1,2,3 << 4.
        for b in &mut block[128..192] {
            *b = 0b1110_0100;
        }
        let mut q = [0i8; QK_K];
        q6k_unpack(&block, &mut q);
        assert_eq!(q[0], 5 - 32); // group 0: + (0 << 4)
        assert_eq!(q[32], 5 + 16 - 32); // group 1: + (1 << 4)
        assert_eq!(q[64], 5 + 32 - 32); // group 2
        assert_eq!(q[96], 5 + 48 - 32); // group 3
        assert_eq!(q[128 + 64], 5 + 32 - 32); // second half behaves alike
    }

    #[test]
    fn q6k_f32_matches_f64_oracle() {
        let mut rng = SplitMix64(7);
        let (rows, cols) = (3, 512);
        let mut weights = vec![0u8; rows * cols / QK_K * Q6K_BLOCK_BYTES];
        for chunk in weights.chunks_mut(Q6K_BLOCK_BYTES) {
            for b in chunk[..208].iter_mut() {
                *b = (rng.next_u64() & 0xff) as u8;
            }
            chunk[208..210].copy_from_slice(&0x2C00u16.to_le_bytes()); // d = 0.0625
        }
        let x: Vec<f32> = (0..cols).map(|_| rng.next_signed_unit()).collect();
        let got = matvec_q6k_f32(&weights, rows, cols, &x).unwrap();
        let oracle = matvec_q6k_f64(&weights, rows, cols, &x).unwrap();
        assert!(max_rel_err(&got, &oracle).unwrap() < 1e-4);
        assert!(matvec_q6k_f32(&weights[1..], rows, cols, &x).is_err());
    }
    #[test]
    fn q8_quantization_round_trips_within_half_a_step() {
        let mut rng = SplitMix64(11);
        let x: Vec<f32> = (0..96).map(|_| rng.next_signed_unit() * 3.0).collect();
        let q = quantize_q8(&x);
        for (i, v) in x.iter().enumerate() {
            let back = q.q[i] as f32 * q.d[i / 32];
            assert!((back - v).abs() <= q.d[i / 32] * 0.5001, "element {i}");
        }
        for g in 0..3 {
            let sum: i32 = q.q[g * 32..(g + 1) * 32].iter().map(|&v| v as i32).sum();
            assert!((q.s[g] - q.d[g] * sum as f32).abs() < 1e-5);
        }
        let zero = quantize_q8(&[0.0; 32]);
        assert!(zero.q.iter().all(|&v| v == 0) && zero.d[0] == 0.0);
    }
    #[test]
    fn f16_conversion_round_trips_and_rounds_to_even() {
        for h in [0u16, 1, 0x3c00, 0xbc00, 0x7bff, 0x0400, 0x03ff, 0x3555] {
            assert_eq!(f32_to_f16_bits(f16_to_f32(h)), h, "{h:#06x}");
        }
        assert_eq!(f32_to_f16_bits(1.0 + 1.0 / 2048.0), 0x3c00); // tie rounds to even
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f32_to_f16_bits(1e9), 0x7c00);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    }}
