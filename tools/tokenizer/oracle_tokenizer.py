#!/usr/bin/env python3
"""Independent tokenizer oracle: HF tokenizers (Rust, installed) BPE
instantiated from GGUF-exported vocab/merges, implementing the official
gemma4 semantics (space->U+2581 escape, newline-only pre-split,
byte_fallback <0xXX>, forced BOS, specials partitioned as added tokens).

Usage: oracle_tokenizer.py <tokenizer-export.json> <text>

Adapter for reference-only semantics (llama.cpp llama-vocab.cpp @
a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd, semantics only): merge
halves/outputs that are not final vocab pieces get VIRTUAL ids >= vocab
size; after encoding, virtual ids are expanded through the original
byte fallback and asserted to never leak into model-visible ids. Only
entries with no separator at byte offset >= 1 are dropped (the
reference stores them as empty pairs that can never match a symbol).
"""
import json, sys
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, AddedToken, Regex


def build(export_path):
    d = json.load(open(export_path))
    pieces, types, merges_raw = d["tokens"], d["types"], d["merges"]
    bos = d["bos_token_id"]
    nv = len(pieces)

    vocab = {}
    for i, p in enumerate(pieces):
        vocab[p] = i  # last wins (matches reference ascending insertion)

    virtual = {}

    def vid(piece):
        if piece not in vocab:
            vocab[piece] = nv + len(virtual)
            virtual[vocab[piece]] = piece
        return piece

    parsed = []
    for m in merges_raw:
        pos = m.find(" ", 1)
        if pos == -1:
            continue  # reference: empty pair, can never match a symbol
        parsed.append((m[:pos], m[pos + 1:]))
    for left, right in parsed:
        vid(left); vid(right); vid(left + right)
    merges = list(parsed)  # original rank order preserved

    tok = Tokenizer(models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token=pieces[d["unknown_token_id"]],
        byte_fallback=True,
        fuse_unk=False,
    ))
    tok.normalizer = normalizers.Replace(" ", "\u2581")
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\n+"), behavior="isolated",
                                             invert=False)
    for i, t in enumerate(types):
        if t in (2, 3, 4):
            tok.add_tokens([AddedToken(pieces[i], single_word=False,
                                       normalized=False, special=(t != 4))])

    def encode(text):
        ids = tok.encode(text, add_special_tokens=False).ids
        out = []
        for t in ids:
            if t in virtual:
                for b in virtual[t].encode("utf-8"):
                    out.append(vocab[f"<0x{b:02X}>"])
            else:
                out.append(t)
        assert all(0 <= t < nv for t in out), "virtual id leaked to model"
        return [bos] + out

    return encode


if __name__ == "__main__":
    print(build(sys.argv[1])(sys.argv[2]))
