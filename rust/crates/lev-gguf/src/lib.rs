//! Minimal GGUF v3 reader: metadata, tensor directory and quantization
//! layouts. Every count and length is bounded before allocation, tensor
//! extents are checked against the file length, and no weight bytes are read.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub const MAX_KV: u64 = 1 << 16;
pub const MAX_TENSORS: u64 = 1 << 20;
pub const MAX_STRING: u64 = 1 << 24;
pub const MAX_DIMS: u32 = 8;
/// Upper bound for one `read_tensor_bytes` call.
pub const MAX_READ: u64 = 1 << 28;
/// Arrays up to this length are kept; longer ones (token tables) are skipped.
pub const KEPT_ARRAY_LEN: u64 = 64;
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    Bad(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io error: {e}"),
            GgufError::Bad(m) => write!(f, "invalid gguf: {m}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        GgufError::Io(e)
    }
}

fn bad<T>(msg: impl Into<String>) -> Result<T, GgufError> {
    Err(GgufError::Bad(msg.into()))
}

/// ggml tensor element types. Quantization layouts are data: block element
/// count and block byte size, from which sizes (and later oracles and kernel
/// decoders) are derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::Bf16,
            other => Self::Unknown(other),
        }
    }

    /// `(elements per block, bytes per block)`; `None` for unknown types.
    pub fn block_layout(self) -> Option<(u64, u64)> {
        Some(match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::Unknown(_) => return None,
        })
    }

    pub fn name(self) -> String {
        match self {
            Self::Q4_0 => "Q4_0".into(),
            Self::Q4_1 => "Q4_1".into(),
            Self::Q5_0 => "Q5_0".into(),
            Self::Q5_1 => "Q5_1".into(),
            Self::Q8_0 => "Q8_0".into(),
            Self::Q8_1 => "Q8_1".into(),
            Self::Q2K => "Q2_K".into(),
            Self::Q3K => "Q3_K".into(),
            Self::Q4K => "Q4_K".into(),
            Self::Q5K => "Q5_K".into(),
            Self::Q6K => "Q6_K".into(),
            Self::Q8K => "Q8_K".into(),
            Self::F32 => "F32".into(),
            Self::F16 => "F16".into(),
            Self::Bf16 => "BF16".into(),
            Self::Unknown(id) => format!("type{id}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// `values` is `None` when the array was longer than [`KEPT_ARRAY_LEN`].
    Array { elem_type: u32, len: u64, values: Option<Vec<Value>> },
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> Option<u64> {
        self.dims.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d))
    }

    /// Byte size from the block layout; `None` on unknown type, overflow, or
    /// an element count that is not a whole number of blocks.
    pub fn byte_size(&self) -> Option<u64> {
        let (block_elems, block_bytes) = self.ty.block_layout()?;
        let n = self.element_count()?;
        if n % block_elems != 0 {
            return None;
        }
        (n / block_elems).checked_mul(block_bytes)
    }
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute offset of the tensor data section.
    pub data_offset: u64,
    pub file_len: u64,
    /// Arrays longer than [`KEPT_ARRAY_LEN`] are not kept in `metadata`: key -> (element type,
    /// length, absolute offset of the first element), for [`GgufFile::read_string_array`] and friends.
    pub skipped_arrays: BTreeMap<String, (u32, u64, u64)>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        let file_len = file.metadata()?.len();
        parse(BufReader::with_capacity(1 << 16, file), file_len)
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(Value::as_u64)
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(Value::as_str)
    }

    /// Reads a long array of strings (such as `tokenizer.ggml.tokens`) from the file.
    pub fn read_string_array<F: Read + Seek>(&self, file: &mut F, key: &str) -> Result<Vec<String>, GgufError> {
        let Some(&(ty, len, off)) = self.skipped_arrays.get(key) else {
            return bad(format!("{key}: not a skipped array"));
        };
        if ty != 8 {
            return bad(format!("{key}: not a string array"));
        }
        file.seek(SeekFrom::Start(off))?;
        let mut r = io::BufReader::with_capacity(1 << 20, file);
        let mut scratch = Vec::new();
        let mut out = Vec::with_capacity(len as usize);
        for _ in 0..len {
            out.push(read_string(&mut r, &mut scratch)?);
        }
        Ok(out)
    }

    /// Reads a long array of 32-bit integers (such as `tokenizer.ggml.token_type`).
    pub fn read_i32_array<F: Read + Seek>(&self, file: &mut F, key: &str) -> Result<Vec<i32>, GgufError> {
        let Some(&(ty, len, off)) = self.skipped_arrays.get(key) else {
            return bad(format!("{key}: not a skipped array"));
        };
        if ty != 4 && ty != 5 {
            return bad(format!("{key}: not a 32-bit integer array"));
        }
        file.seek(SeekFrom::Start(off))?;
        let mut bytes = vec![0u8; len as usize * 4];
        file.read_exact(&mut bytes)?;
        Ok(bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Reads `len` bytes starting `start` bytes into the data of tensor `t`.
    /// The range is checked against the tensor extent before any I/O.
    pub fn read_tensor_bytes<F: Read + Seek>(
        &self,
        file: &mut F,
        t: &TensorInfo,
        start: u64,
        len: u64,
    ) -> Result<Vec<u8>, GgufError> {
        let Some(size) = t.byte_size() else {
            return bad(format!("tensor {}: unknown size", t.name));
        };
        match start.checked_add(len) {
            Some(end) if len <= MAX_READ && end <= size => {}
            _ => return bad(format!("tensor {}: read range out of bounds", t.name)),
        }
        file.seek(SeekFrom::Start(self.data_offset + t.offset + start))?;
        let mut buf = vec![0u8; len as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Sum of all tensor byte sizes.
    pub fn total_tensor_bytes(&self) -> Option<u64> {
        self.tensors.iter().try_fold(0u64, |acc, t| acc.checked_add(t.byte_size()?))
    }
}

struct Counted<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

fn read_array<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N], GgufError> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    Ok(u32::from_le_bytes(read_array(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    Ok(u64::from_le_bytes(read_array(r)?))
}

fn read_string<R: Read>(r: &mut R, scratch: &mut Vec<u8>) -> Result<String, GgufError> {
    let len = read_u64(r)?;
    if len > MAX_STRING {
        return bad(format!("string length {len} exceeds {MAX_STRING}"));
    }
    scratch.clear();
    scratch.resize(len as usize, 0);
    r.read_exact(scratch)?;
    String::from_utf8(scratch.clone()).or_else(|_| bad("string is not valid utf-8"))
}

fn skip_string<R: Read>(r: &mut R, scratch: &mut Vec<u8>) -> Result<(), GgufError> {
    let len = read_u64(r)?;
    if len > MAX_STRING {
        return bad(format!("string length {len} exceeds {MAX_STRING}"));
    }
    scratch.clear();
    scratch.resize(len as usize, 0);
    r.read_exact(scratch)?;
    Ok(())
}

fn scalar_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn read_scalar<R: Read>(r: &mut R, ty: u32, scratch: &mut Vec<u8>) -> Result<Value, GgufError> {
    Ok(match ty {
        0 => Value::U8(read_array::<1, _>(r)?[0]),
        1 => Value::I8(read_array::<1, _>(r)?[0] as i8),
        2 => Value::U16(u16::from_le_bytes(read_array(r)?)),
        3 => Value::I16(i16::from_le_bytes(read_array(r)?)),
        4 => Value::U32(u32::from_le_bytes(read_array(r)?)),
        5 => Value::I32(i32::from_le_bytes(read_array(r)?)),
        6 => Value::F32(f32::from_le_bytes(read_array(r)?)),
        7 => Value::Bool(read_array::<1, _>(r)?[0] != 0),
        8 => Value::Str(read_string(r, scratch)?),
        10 => Value::U64(u64::from_le_bytes(read_array(r)?)),
        11 => Value::I64(i64::from_le_bytes(read_array(r)?)),
        12 => Value::F64(f64::from_le_bytes(read_array(r)?)),
        other => return bad(format!("unsupported value type {other}")),
    })
}

fn read_value<R: Read>(r: &mut R, ty: u32, scratch: &mut Vec<u8>) -> Result<Value, GgufError> {
    if ty != 9 {
        return read_scalar(r, ty, scratch);
    }
    let elem_type = read_u32(r)?;
    let len = read_u64(r)?;
    if elem_type == 9 {
        return bad("nested arrays are not supported");
    }
    if len <= KEPT_ARRAY_LEN {
        let mut values = Vec::with_capacity(len as usize);
        for _ in 0..len {
            values.push(read_scalar(r, elem_type, scratch)?);
        }
        return Ok(Value::Array { elem_type, len, values: Some(values) });
    }
    if elem_type == 8 {
        for _ in 0..len {
            skip_string(r, scratch)?;
        }
    } else {
        let Some(size) = scalar_size(elem_type) else {
            return bad(format!("unsupported array element type {elem_type}"));
        };
        let Some(total) = len.checked_mul(size) else {
            return bad("array byte length overflows");
        };
        let copied = io::copy(&mut r.by_ref().take(total), &mut io::sink())?;
        if copied != total {
            return bad("truncated array");
        }
    }
    Ok(Value::Array { elem_type, len, values: None })
}

pub fn parse<R: Read>(reader: R, file_len: u64) -> Result<GgufFile, GgufError> {
    let mut r = Counted { inner: reader, pos: 0 };
    let magic: [u8; 4] = read_array(&mut r)?;
    if &magic != b"GGUF" {
        return bad("missing GGUF magic");
    }
    let version = read_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        return bad(format!("unsupported version {version}"));
    }
    let tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if tensor_count > MAX_TENSORS {
        return bad(format!("tensor count {tensor_count} exceeds {MAX_TENSORS}"));
    }
    if kv_count > MAX_KV {
        return bad(format!("metadata count {kv_count} exceeds {MAX_KV}"));
    }

    let mut scratch = Vec::new();
    let mut metadata = BTreeMap::new();
    let mut skipped_arrays = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_string(&mut r, &mut scratch)?;
        let ty = read_u32(&mut r)?;
        // An array value starts with its element type (4 bytes) and length (8 bytes).
        let elements_at = r.pos + 12;
        let value = read_value(&mut r, ty, &mut scratch)?;
        if let Value::Array { elem_type, len, values: None } = &value {
            skipped_arrays.insert(key.clone(), (*elem_type, *len, elements_at));
        }
        if metadata.insert(key.clone(), value).is_some() {
            return bad(format!("duplicate metadata key {key}"));
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut r, &mut scratch)?;
        let n_dims = read_u32(&mut r)?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return bad(format!("tensor {name}: bad dimension count {n_dims}"));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut r)?);
        }
        let ty = GgmlType::from_id(read_u32(&mut r)?);
        let offset = read_u64(&mut r)?;
        tensors.push(TensorInfo { name, dims, ty, offset });
    }

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(v) => match v.as_u64() {
            Some(a) if a > 0 && a.is_power_of_two() => a,
            _ => return bad("general.alignment is not a positive power of two"),
        },
    };
    let data_offset = r.pos.div_ceil(alignment) * alignment;
    if data_offset > file_len {
        return bad("tensor data section starts past end of file");
    }

    for t in &tensors {
        let Some(size) = t.byte_size() else {
            return bad(format!("tensor {}: unsupported or inconsistent size ({:?})", t.name, t.ty));
        };
        if t.offset % alignment != 0 {
            return bad(format!("tensor {}: offset {} not aligned to {alignment}", t.name, t.offset));
        }
        let end = data_offset.checked_add(t.offset).and_then(|s| s.checked_add(size));
        match end {
            Some(e) if e <= file_len => {}
            _ => return bad(format!("tensor {}: extent exceeds file length", t.name)),
        }
    }

    Ok(GgufFile { version, metadata, tensors, data_offset, file_len, skipped_arrays })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// One metadata string, one u32, and one Q4_K tensor of 512 elements.
    fn tiny_gguf(tensor_ty: u32, dims: &[u64], data_bytes: usize) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensors
        b.extend_from_slice(&2u64.to_le_bytes()); // kv
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut b, "test");
        put_str(&mut b, "test.block_count");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes());
        put_str(&mut b, "blk.0.w");
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&tensor_ty.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(std::iter::repeat(0u8).take(data_bytes));
        b
    }

    #[test]
    fn parses_tiny_file() {
        let bytes = tiny_gguf(12, &[256, 2], 2 * 144);
        let g = parse(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.meta_str("general.architecture"), Some("test"));
        assert_eq!(g.meta_u64("test.block_count"), Some(7));
        assert_eq!(g.tensors[0].ty, GgmlType::Q4K);
        assert_eq!(g.tensors[0].byte_size(), Some(288));
        assert_eq!(g.data_offset % 32, 0);
    }

    #[test]
    fn rejects_truncated_tensor_data() {
        let bytes = tiny_gguf(12, &[256, 2], 100);
        assert!(parse(&bytes[..], bytes.len() as u64).is_err());
    }

    #[test]
    fn rejects_partial_block() {
        let bytes = tiny_gguf(12, &[100], 1000);
        assert!(parse(&bytes[..], bytes.len() as u64).is_err());
    }

    #[test]
    fn rejects_bad_magic_and_huge_counts() {
        assert!(parse(&b"NOPE"[..], 4).is_err());
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse(&b[..], b.len() as u64).is_err());
    }

    #[test]
    fn layouts_match_known_block_sizes() {
        assert_eq!(GgmlType::Q4_0.block_layout(), Some((32, 18)));
        assert_eq!(GgmlType::Q4K.block_layout(), Some((256, 144)));
        assert_eq!(GgmlType::Q6K.block_layout(), Some((256, 210)));
    }
}
