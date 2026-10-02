#!/bin/bash
# Cross-check the Leviathan Swift gemma4 tokenizer against the
# independent Rust tokenizers oracle on diverse cases (ASCII, Korean,
# NFD/NFC, emoji+skin-tone, BOM, CRLF, special-token overlaps, full
# canonical chat prompt). Full ID lists must be identical.
#
# Usage: crosscheck.sh <gguf-file> <cli-binary> [workdir]
# Exports tokenizer metadata via the CLI into the workdir first.
set -euo pipefail
GG="$1"; CLI="$2"; WORK="${3:-$(mktemp -d)}"
EXPORT="$WORK/tokenizer-export.json"
"$CLI" gguf-tokenizer-export --file "$GG" --out "$EXPORT" >/dev/null
HERE="$(cd "$(dirname "$0")" && pwd)"
GG="$GG" CLI="$CLI" EXPORT="$EXPORT" ORACLE="$HERE/oracle_tokenizer.py" python3 - <<'PYEOF'
import os, subprocess, sys, importlib.util
spec = importlib.util.spec_from_file_location("oracle", os.environ["ORACLE"])
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
encode = oracle.build(os.environ["EXPORT"])

cases = [
 "Hello",
 "Hello, how are you?",
 "The quick brown fox jumps over the lazy dog 12345!",
 "안녕하세요, 오늘 날씨가 어떤가요?",
 "한국어 문장 테스트: 인공지능과 자연어 처리",
 "emoji 👍🏽 test café naïve",
 "line one\nline two\n\nline three after blank",
 "  leading and trailing spaces   ",
 "tabs\tand\nnewlines\r\nmixed \ufeffBOM",
 "e\u0301 composing vs é precomposed",
 "special overlap: <|turn> embedded <turn|> in text <bos> <eos>",
 "digits 3.14159 and 1,000,000 numbers",
 "<|turn>user\nWrite a haiku about rain.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
]
ok = 0
for t in cases:
    out = subprocess.run([os.environ["CLI"], "gemma-tokenize", "--file",
                          os.environ["GG"], "--text", t],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print("CLI FAIL", repr(t[:40]), out.stderr.strip()); continue
    swift_ids = list(map(int, out.stdout.splitlines()[0].split()))
    rust_ids = encode(t)
    match = swift_ids == rust_ids
    ok += match
    print(("MATCH" if match else "DIFF "), repr(t[:40]))
    if not match:
        print("  swift:", swift_ids); print("  rust :", rust_ids)
print(f"{ok}/{len(cases)} match")
sys.exit(0 if ok == len(cases) else 1)
PYEOF
