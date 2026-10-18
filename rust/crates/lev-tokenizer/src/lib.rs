//! Byte-level BPE tokenizer (GPT-2 style) built from the vocabulary, token types and merge rules
//! stored in a GGUF file, with the `qwen35` pre-tokenizer.
//!
//! Encoding: special tokens in the text are cut out first, the rest is split into pre-tokens by
//! the pre-tokenizer rules, every pre-token is mapped to the byte-level alphabet and merged by
//! rank. Decoding concatenates the bytes of the tokens. The result is checked against
//! llama.cpp's `/tokenize` on a corpus (tools/bench/tokenizer-check.ps1).

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;

pub mod pretok;
mod unicode_tables;

/// Token types as stored in `tokenizer.ggml.token_type`.
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;

#[derive(Debug)]
pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Token bytes (after undoing the byte-level alphabet), for decoding.
    bytes: Vec<Vec<u8>>,
    /// Special tokens, longest first: (text, id, is_control).
    specials: Vec<(String, u32, bool)>,
    byte_to_char: [char; 256],
    pub eos: Option<u32>,
}

/// The GPT-2 byte-to-unicode table: printable bytes map to themselves, the others to U+0100 and up.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut keep: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    let mut back = HashMap::new();
    for b in 0u32..256 {
        let c = if keep.contains(&b) {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
        table[b as usize] = c;
        back.insert(c, b as u8);
    }
    keep.clear();
    (table, back)
}

impl Tokenizer {
    pub fn from_gguf(path: &str) -> Result<Tokenizer, String> {
        let g = lev_gguf::GgufFile::open(path).map_err(|e| e.to_string())?;
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        if g.meta_str("tokenizer.ggml.model") != Some("gpt2") {
            return Err("only gpt2 (byte-level BPE) tokenizers are supported".into());
        }
        let tokens = g.read_string_array(&mut f, "tokenizer.ggml.tokens").map_err(|e| e.to_string())?;
        let types = g.read_i32_array(&mut f, "tokenizer.ggml.token_type").map_err(|e| e.to_string())?;
        let merges = g.read_string_array(&mut f, "tokenizer.ggml.merges").map_err(|e| e.to_string())?;
        let eos = g.meta_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        Tokenizer::new(tokens, types, merges, eos)
    }

    pub fn new(tokens: Vec<String>, types: Vec<i32>, merges: Vec<String>, eos: Option<u32>) -> Result<Tokenizer, String> {
        let (byte_to_char, char_to_byte) = byte_alphabet();
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let mut rules = HashMap::new();
        for (rank, m) in merges.iter().enumerate() {
            let (a, b) = m.split_once(' ').ok_or(format!("bad merge rule {m:?}"))?;
            let (Some(&ia), Some(&ib)) = (ids.get(a), ids.get(b)) else { continue };
            let joined = format!("{a}{b}");
            let Some(&id) = ids.get(&joined) else { continue };
            rules.entry((ia, ib)).or_insert((rank as u32, id));
        }
        // Decoding bytes: ordinary tokens are in the byte alphabet, special tokens are literal text.
        let mut bytes = Vec::with_capacity(tokens.len());
        let mut specials = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let ty = types.get(i).copied().unwrap_or(1);
            if ty == TYPE_CONTROL || ty == TYPE_USER_DEFINED {
                bytes.push(t.as_bytes().to_vec());
                specials.push((t.clone(), i as u32, ty == TYPE_CONTROL));
            } else {
                bytes.push(t.chars().map(|c| char_to_byte.get(&c).copied().unwrap_or(b'?')).collect());
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        Ok(Tokenizer { tokens, ids, merges: rules, bytes, specials, byte_to_char, eos })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Token ids of `text`. With `parse_special`, special tokens written in the text (such as
    /// `<|im_start|>`) become their ids; without it only user-defined ones do.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            // Find the earliest special token occurrence (longest at equal position).
            let mut best: Option<(usize, usize, u32)> = None;
            for (s, id, control) in &self.specials {
                if *control && !parse_special {
                    continue;
                }
                if let Some(at) = rest.find(s.as_str()) {
                    if best.is_none_or(|(b, _, _)| at < b) {
                        best = Some((at, s.len(), *id));
                    }
                }
            }
            match best {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                    continue 'outer;
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in pretok::split(text) {
            let mapped: Vec<char> = piece.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            self.bpe(&mapped, out);
        }
    }

    /// Merges one pre-token (already in the byte alphabet) by rank.
    fn bpe(&self, chars: &[char], out: &mut Vec<u32>) {
        if chars.is_empty() {
            return;
        }
        // Whole pre-token is a token: the common case.
        let whole: String = chars.iter().collect();
        if let Some(&id) = self.ids.get(&whole) {
            out.push(id);
            return;
        }
        let mut sym: Vec<(u32, usize, usize, bool)> = Vec::with_capacity(chars.len()); // (id, prev, next, alive)
        for (i, c) in chars.iter().enumerate() {
            let id = self.ids.get(&c.to_string()).copied().unwrap_or(0);
            sym.push((id, i.wrapping_sub(1), i + 1, true));
        }
        let n = sym.len();
        sym[n - 1].2 = usize::MAX;
        // Heap of candidate merges: lowest rank first, then leftmost.
        let mut heap: BinaryHeap<Reverse<(u32, usize, u32, u32)>> = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Reverse<(u32, usize, u32, u32)>>, sym: &Vec<(u32, usize, usize, bool)>, l: usize| {
            let r = sym[l].2;
            if r == usize::MAX {
                return;
            }
            if let Some(&(rank, _)) = self.merges.get(&(sym[l].0, sym[r].0)) {
                heap.push(Reverse((rank, l, sym[l].0, sym[r].0)));
            }
        };
        for l in 0..n - 1 {
            push(&mut heap, &sym, l);
        }
        while let Some(Reverse((_, l, lid, rid))) = heap.pop() {
            let r = sym[l].2;
            // Skip stale candidates (a symbol was merged since).
            if !sym[l].3 || r == usize::MAX || !sym[r].3 || sym[l].0 != lid || sym[r].0 != rid {
                continue;
            }
            let &(_, merged) = self.merges.get(&(lid, rid)).unwrap();
            sym[l].0 = merged;
            sym[l].2 = sym[r].2;
            sym[r].3 = false;
            if sym[r].2 != usize::MAX {
                let nr = sym[r].2;
                sym[nr].1 = l;
            }
            if sym[l].1 != usize::MAX {
                push(&mut heap, &sym, sym[l].1);
            }
            push(&mut heap, &sym, l);
        }
        let mut i = 0;
        while i != usize::MAX {
            out.push(sym[i].0);
            i = sym[i].2;
        }
    }

    /// The bytes of one token (the text of a special token, the raw bytes of an ordinary one).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], |v| v.as_slice())
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = Vec::new();
        for &i in ids {
            b.extend_from_slice(self.token_bytes(i));
        }
        String::from_utf8_lossy(&b).into_owned()
    }
}

/// Turns a stream of token bytes into text, holding back an incomplete UTF-8 sequence until the
/// next token completes it.
#[derive(Default, Debug)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                // An incomplete tail (at most 3 bytes) waits; bytes that can never complete are replaced.
                if e.error_len().is_some() {
                    let out = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    out
                } else {
                    self.pending.drain(..valid);
                    out
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny vocabulary: the byte alphabet letters used, plus a few merges.
    fn tiny() -> Tokenizer {
        let (table, _) = byte_alphabet();
        let c = |b: u8| table[b as usize];
        let mut tokens: Vec<String> = Vec::new();
        for b in [b'a', b'b', b'c', b' ', b'<', b'|', b'>'] {
            tokens.push(c(b).to_string());
        }
        let sp = c(b' ');
        let ab = format!("{}{}", c(b'a'), c(b'b'));
        let abc = format!("{ab}{}", c(b'c'));
        let spa = format!("{sp}{}", c(b'a'));
        tokens.extend([ab.clone(), abc.clone(), spa.clone(), "<|end|>".to_string()]);
        let mut types = vec![1; tokens.len()];
        *types.last_mut().unwrap() = 3;
        let merges = vec![format!("{} {}", c(b'a'), c(b'b')), format!("{ab} {}", c(b'c')), format!("{sp} {}", c(b'a'))];
        Tokenizer::new(tokens, types, merges, None).unwrap()
    }

    #[test]
    fn merges_by_rank_and_round_trips() {
        let t = tiny();
        let ids = t.encode("abc ab", false);
        assert_eq!(t.decode(&ids), "abc ab");
        assert_eq!(ids.len(), 3); // abc, space, ab
        assert_eq!(t.encode("ab", false).len(), 1);
    }

    #[test]
    fn special_tokens_are_cut_out_only_when_asked() {
        let t = tiny();
        let with = t.encode("ab<|end|>c", true);
        assert_eq!(with.len(), 3);
        assert_eq!(t.decode(&with), "ab<|end|>c");
        let without = t.encode("ab<|end|>c", false);
        assert!(without.len() > 3); // the marker is ordinary text then
    }

    #[test]
    fn utf8_stream_holds_back_incomplete_sequences() {
        let mut s = Utf8Stream::default();
        let bytes = "한".as_bytes(); // three bytes
        assert_eq!(s.push(&bytes[..1]), "");
        assert_eq!(s.push(&bytes[1..2]), "");
        assert_eq!(s.push(&bytes[2..]), "한");
        assert_eq!(s.push(b"ok"), "ok");
    }
}