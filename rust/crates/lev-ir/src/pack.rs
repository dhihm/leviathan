//! Tensor pack ("LEVT"): named float32 / int64 tensors, see tools/model_ir/make_fixture.py.

use std::collections::BTreeMap;

use crate::tensor::{IntTensor, Tensor};

pub enum Packed {
    F32(Tensor),
    I64(IntTensor),
}

pub fn read(bytes: &[u8]) -> Result<BTreeMap<String, Packed>, String> {
    let mut c = Cursor { b: bytes, i: 0 };
    if c.take(4)? != b"LEVT" {
        return Err("not a LEVT tensor pack".into());
    }
    let count = c.u32()?;
    let mut out = BTreeMap::new();
    for _ in 0..count {
        let nlen = c.u16()? as usize;
        let name = String::from_utf8(c.take(nlen)?.to_vec()).map_err(|e| e.to_string())?;
        let dtype = c.take(1)?[0];
        let rank = c.take(1)?[0] as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(c.u64()? as usize);
        }
        let nbytes = c.u64()? as usize;
        let raw = c.take(nbytes)?;
        let n: usize = shape.iter().product();
        let t = match dtype {
            0 if nbytes == n * 4 => Packed::F32(Tensor { shape, data: raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect() }),
            1 if nbytes == n * 8 => {
                Packed::I64(IntTensor { shape, data: raw.chunks_exact(8).map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])).collect() })
            }
            _ => return Err(format!("tensor {name}: bad dtype {dtype} or byte length {nbytes}")),
        };
        out.insert(name, t);
    }
    Ok(out)
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.i.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("truncated tensor pack")?;
        let s = &self.b[self.i..end];
        self.i = end;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}
