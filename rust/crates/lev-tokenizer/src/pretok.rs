//! The `qwen35` pre-tokenizer: splits text into pre-tokens the way
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//! | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//! | \s*[\r\n]+ | \s+(?!\S) | \s+
//! ```
//!
//! does, with the alternatives tried in order at every position. Unicode classes come from the
//! generated range tables (unicode_tables.rs, from Python's `unicodedata`); only `\s` uses the
//! standard library. Checked against llama.cpp on a multilingual corpus (tools/bench/tokenizer-check.ps1).

use crate::unicode_tables::{LETTERS, MARKS, NUMBERS};

fn in_ranges(table: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    table.binary_search_by(|&(lo, hi)| if cp < lo { std::cmp::Ordering::Greater } else if cp > hi { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal }).is_ok()
}

/// `\p{L}` or `\p{M}`.
fn is_letter_or_mark(c: char) -> bool {
    in_ranges(LETTERS, c) || in_ranges(MARKS, c)
}

fn is_letter(c: char) -> bool {
    in_ranges(LETTERS, c)
}

fn is_number(c: char) -> bool {
    in_ranges(NUMBERS, c)
}
fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// `[^\s\p{L}\p{M}\p{N}]`.
fn is_symbol(c: char) -> bool {
    !c.is_whitespace() && !is_letter_or_mark(c) && !is_number(c)
}

pub fn split(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let at = |i: usize| chars.get(i).map(|&(_, c)| c);
    let byte = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i].1;
        let mut end = None;

        // 1. contractions
        if c == '\'' {
            let low = |k: usize| at(i + k).map(|x| x.to_ascii_lowercase());
            end = match (low(1), low(2)) {
                (Some('s' | 't' | 'm' | 'd'), _) => Some(i + 2),
                (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => Some(i + 3),
                _ => None,
            };
        }
        // 2. optional prefix, then letters and marks
        if end.is_none() {
            let prefix_ok = |x: char| !is_newline(x) && !is_letter(x) && !is_number(x);
            if prefix_ok(c) && at(i + 1).is_some_and(is_letter_or_mark) {
                let mut j = i + 1;
                while at(j).is_some_and(is_letter_or_mark) {
                    j += 1;
                }
                end = Some(j);
            } else if is_letter_or_mark(c) {
                let mut j = i;
                while at(j).is_some_and(is_letter_or_mark) {
                    j += 1;
                }
                end = Some(j);
            }
        }
        // 3. one number
        if end.is_none() && is_number(c) {
            end = Some(i + 1);
        }
        // 4. optional space, symbols, newlines
        if end.is_none() {
            let start = if c == ' ' && at(i + 1).is_some_and(is_symbol) { i + 1 } else { i };
            if at(start).is_some_and(is_symbol) {
                let mut j = start;
                while at(j).is_some_and(is_symbol) {
                    j += 1;
                }
                while at(j).is_some_and(is_newline) {
                    j += 1;
                }
                end = Some(j);
            }
        }
        // 5-7. whitespace
        if end.is_none() && c.is_whitespace() {
            let mut run_end = i;
            while at(run_end).is_some_and(char::is_whitespace) {
                run_end += 1;
            }
            let last_newline = (i..run_end).rev().find(|&k| at(k).is_some_and(is_newline));
            end = Some(if let Some(k) = last_newline {
                k + 1 // \s*[\r\n]+ ends after the last newline of the run
            } else if run_end == n {
                run_end // \s+(?!\S) at the end of the text takes everything
            } else if run_end - i >= 2 {
                run_end - 1 // \s+(?!\S): leave the last space for the next token
            } else {
                run_end // \s+
            });
        }
        let end = end.unwrap_or(i + 1);
        out.push(&text[byte(i)..byte(end)]);
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::split;

    #[test]
    fn splits_words_numbers_and_punctuation() {
        assert_eq!(split("Hello, world!"), ["Hello", ",", " world", "!"]);
        assert_eq!(split("abc123"), ["abc", "1", "2", "3"]);
        assert_eq!(split("it's I'M don't"), ["it", "'s", " I", "'M", " don", "'t"]);
    }

    #[test]
    fn whitespace_rules() {
        assert_eq!(split("a  b"), ["a", " ", " b"]); // \s+(?!\S) leaves one space for the word
        assert_eq!(split("a\n\nb"), ["a", "\n\n", "b"]);
        assert_eq!(split("a \nb"), ["a", " \n", "b"]);
        assert_eq!(split("a  "), ["a", "  "]);
        assert_eq!(split("x = 1;\n"), ["x", " =", " ", "1", ";\n"]);
    }

    #[test]
    fn non_ascii_text() {
        assert_eq!(split("안녕 세상"), ["안녕", " 세상"]);
        assert_eq!(split("é"), ["é"]);
    }

    #[test]
    fn pieces_cover_the_text() {
        let s = "Mixed 한글 text,\twith\r\nodd   spacing  and 😀 emoji!";
        assert_eq!(split(s).concat(), s);
    }
}