//! Dequantization of whole GGUF tensors to f32, for feeding the IR reference interpreter.

use std::fs::File;

use lev_cpu::{dequant_block_f32, dequant_q6k_block_f32, f16_to_f32, Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES, QK_K};
use lev_gguf::{GgmlType, GgufFile};

use crate::Error;

/// Elements of a tensor (or of `rows` rows of it) as f32, in GGUF memory order. For 2-D weights
/// this is the row-major order of the PyTorch `[out, in]` layout.
pub fn dequantize(g: &GgufFile, file: &mut File, name: &str, first_row: usize, rows: Option<usize>) -> Result<Vec<f32>, Error> {
    let t = g.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
    let cols = t.dims[0] as usize;
    let total_rows = t.dims.iter().skip(1).product::<u64>().max(1) as usize;
    let n_rows = rows.unwrap_or(total_rows - first_row);
    if first_row + n_rows > total_rows {
        return Err(format!("{name}: rows {first_row}..{} out of {total_rows}", first_row + n_rows).into());
    }
    let (block_elems, block_bytes) = match t.ty {
        GgmlType::F32 => (1, 4),
        GgmlType::F16 => (1, 2),
        GgmlType::Q4K => (QK_K, Q4K_BLOCK_BYTES),
        GgmlType::Q6K => (QK_K, Q6K_BLOCK_BYTES),
        other => return Err(format!("{name}: cannot dequantize {}", other.name()).into()),
    };
    if cols % block_elems != 0 {
        return Err(format!("{name}: row of {cols} elements is not a whole number of blocks").into());
    }
    let row_bytes = cols / block_elems * block_bytes;
    let bytes = g.read_tensor_bytes(file, t, (first_row * row_bytes) as u64, (n_rows * row_bytes) as u64)?;
    let mut out = Vec::with_capacity(n_rows * cols);
    match t.ty {
        GgmlType::F32 => out.extend(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
        GgmlType::F16 => out.extend(bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))),
        GgmlType::Q4K => {
            let mut tmp = [0f32; QK_K];
            for b in bytes.chunks_exact(Q4K_BLOCK_BYTES) {
                dequant_block_f32(b, &mut tmp);
                out.extend_from_slice(&tmp);
            }
        }
        GgmlType::Q6K => {
            let mut tmp = [0f32; QK_K];
            for b in bytes.chunks_exact(Q6K_BLOCK_BYTES) {
                dequant_q6k_block_f32(b, &mut tmp);
                out.extend_from_slice(&tmp);
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}
