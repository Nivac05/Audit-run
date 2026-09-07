#!/usr/bin/env python3
"""
Tokenizer registry.

Specs accepted by `load(spec)`:

    gpt2            real OpenAI GPT-2 byte-level BPE, loaded OFFLINE from
                    partA/vendor/{encoder.json,vocab.bpe} by seeding tiktoken's
                    on-disk cache. Verified against known token IDs at import time
                    by tests/test_tokenizers.py.
    bytes           UTF-8 byte tokenizer. Not a real deployment target; it is the
                    theoretical FLOOR for any byte-level BPE (1 token per byte).
                    Used as a reference point in the byte-fallback experiment.
    spm:<path>      a locally trained SentencePiece model (see train_spm.py)
    hf:<repo_id>    any HuggingFace tokenizer, e.g. hf:xlm-roberta-base,
                    hf:ai4bharat/IndicBERTv2-MLM-only. REQUIRES NETWORK.

Every loader returns a callable str -> list[int].
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil

VENDOR = pathlib.Path(__file__).resolve().parent / "vendor"
_CACHE = pathlib.Path(__file__).resolve().parent / ".tiktoken_cache"

# The two blobs tiktoken wants for the "gpt2" encoding. We cannot reach
# openaipublic.blob.core.windows.net from the sandbox, so we vendor the files and
# seed the cache under the sha1-of-URL keys tiktoken looks up.
_GPT2_BLOBS = {
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe": "vocab.bpe",
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json": "encoder.json",
}

# sha256 of the canonical OpenAI artifacts. If these do not match, the vendored
# files are not the real GPT-2 vocab and every number downstream is worthless,
# so we fail loudly rather than silently benchmarking something else.
EXPECTED_SHA256 = {
    "encoder.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "vocab.bpe": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
}


def _verify_vendor() -> None:
    for name, want in EXPECTED_SHA256.items():
        p = VENDOR / name
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}. Run tools/fetch_gpt2_vocab.sh to populate partA/vendor/."
            )
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            raise ValueError(f"{name} sha256 mismatch\n  expected {want}\n  got      {got}")


def _seed_tiktoken_cache() -> None:
    _CACHE.mkdir(exist_ok=True)
    for url, name in _GPT2_BLOBS.items():
        key = hashlib.sha1(url.encode()).hexdigest()
        dst = _CACHE / key
        if not dst.exists():
            shutil.copy(VENDOR / name, dst)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_CACHE)


def _load_gpt2():
    _verify_vendor()
    _seed_tiktoken_cache()
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    # Cheap self-check: these IDs are stable, public GPT-2 facts.
    assert enc.encode("hello world") == [31373, 995], "GPT-2 vocab is not the real one"
    return enc.encode


def _load_bytes():
    return lambda s: list(s.encode("utf-8"))


def _load_spm(path: str):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=path)
    return lambda s: sp.encode(s, out_type=int)


def _load_hf(repo: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    # add_special_tokens=False matters: <s>/</s> are a constant per-sentence
    # offset that would otherwise contaminate the per-sentence denominator.
    return lambda s: tok.encode(s, add_special_tokens=False)


def load(spec: str):
    if spec == "gpt2":
        return _load_gpt2()
    if spec == "bytes":
        return _load_bytes()
    if spec.startswith("spm:"):
        return _load_spm(spec[4:])
    if spec.startswith("hf:"):
        return _load_hf(spec[3:])
    raise ValueError(f"unknown tokenizer spec: {spec!r}")


def available(spec: str) -> bool:
    """True if `spec` can actually be loaded here (used to skip network tokenizers)."""
    try:
        load(spec)
        return True
    except Exception:
        return False
