#!/usr/bin/env python3
"""
train_spm.py -- train a small multilingual SentencePiece tokenizer locally.

WHY THIS EXISTS AND WHAT IT IS NOT
  A3 asks for a multilingual/Indic-aware tokenizer. The right ones
  (xlm-roberta-base, IndicBERTv2, sarvam) live on huggingface.co, which the
  audit machine cannot reach. This trains a real BPE tokenizer offline so the
  A3 pipeline has a second, Indic-covering tokenizer end to end.

  IT IS CONTAMINATED. It is trained on the same text it is then measured on,
  so its fertility is an optimistic LOWER BOUND, not an estimate of any
  production tokenizer. It is admissible for exactly one claim -- "a tokenizer
  whose vocabulary covers these scripts does not hit byte fallback" -- and for
  nothing quantitative. Every headline number in A4 comes from gpt2 and from
  the HF tokenizers, not from this.

Usage:
    python train_spm.py --input corpora/smoke --vocab 1000 --out vendor/spm_multi
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import tempfile

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="corpora/smoke")
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--out", default="vendor/spm_multi")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.input}/*.txt"))
    if not files:
        raise SystemExit(f"no .txt under {args.input}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as t:
        for f in files:
            t.write(pathlib.Path(f).read_text(encoding="utf-8"))
        merged = t.name

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=merged,
        model_prefix=args.out,
        vocab_size=args.vocab,
        model_type="bpe",
        character_coverage=1.0,   # must be 1.0: 0.9995 would drop rare Indic glyphs
        byte_fallback=True,       # so unseen characters degrade gracefully, as real ones do
        normalization_rule_name="nfkc",
        train_extremely_large_corpus=False,
    )
    print(f"\nwrote {args.out}.model  (vocab={args.vocab}, trained on {len(files)} files)")
    print("REMINDER: contaminated. Diagnostic only. See docstring.")


if __name__ == "__main__":
    main()
