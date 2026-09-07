#!/usr/bin/env python3
"""
run_a3.py -- corrected cross-language comparison across tokenizers x denominators.

Prints one block per tokenizer: absolute rates and ratios vs English, under five
denominators. Tokenizers that cannot be loaded here (network ones) are reported
as SKIPPED rather than silently omitted or, worse, guessed at.

Usage:
    python run_a3.py --corpus corpora/smoke --langs eng,hin,kan,tam,tel
    python run_a3.py --corpus corpora/flores --langs eng,hin,kan,tam,tel,ben,mar \
                     --tokenizers gpt2,bytes,hf:xlm-roberta-base
"""
from __future__ import annotations

import argparse
import io
import pathlib
import sys

import fertility_v2 as F
import tokenizers_local

HERE = pathlib.Path(__file__).resolve().parent

DEFAULT_TOKENIZERS = [
    "gpt2",                 # the tokenizer REPORT_v0 measured
    "bytes",                # theoretical floor for byte-level BPE
    "hf:xlm-roberta-base",  # multilingual, 250k vocab, Indic-aware  (network)
    "hf:ai4bharat/IndicBERTv2-MLM-only",  # Indic-specialised          (network)
]


def block(tokname, encode, corpus: pathlib.Path, langs, out):
    p = lambda *a: print(*a, file=out)
    res = {}
    counts = set()
    for lang in langs:
        path = corpus / f"{lang}.txt"
        if not path.exists():
            p(f"  (missing {path}, skipping {lang})")
            continue
        lines = F.read_lines(str(path))
        counts.add(len(lines))
        res[lang] = F.measure(lang, lines, encode)

    parallel = len(counts) == 1
    p(f"\n### tokenizer: {tokname}")
    if not parallel:
        p(f"  !! line counts differ {sorted(counts)} -- NOT parallel, tok/sent is invalid")
    p(F.report(res, "eng", parallel))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpora/smoke")
    ap.add_argument("--langs", default="eng,hin,kan,tam,tel")
    ap.add_argument("--tokenizers", default=",".join(DEFAULT_TOKENIZERS))
    ap.add_argument("--out", default="results/a3_table.txt")
    args = ap.parse_args()

    corpus = pathlib.Path(args.corpus)
    langs = args.langs.split(",")
    buf = io.StringIO()
    p = lambda *a: print(*a, file=buf)

    p("=" * 88)
    p(f"A3 CORRECTED ANALYSIS   corpus={corpus}  langs={','.join(langs)}")
    p("=" * 88)
    if "smoke" in str(corpus):
        p("!! corpora/smoke is a 12-sentence author-written SMOKE TEST, not an eval set.")
        p("!! Numbers below prove the pipeline runs. Do not quote them. Run build_corpus.py.")

    skipped = []
    for spec in args.tokenizers.split(","):
        try:
            encode = tokenizers_local.load(spec)
        except Exception as e:
            skipped.append((spec, f"{type(e).__name__}: {str(e)[:90]}"))
            continue
        block(spec, encode, corpus, langs, buf)

    if skipped:
        p("")
        p("SKIPPED (could not load here -- almost certainly no network):")
        for spec, why in skipped:
            p(f"  {spec:<38} {why}")

    text = buf.getvalue()
    print(text)
    outp = HERE / args.out
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
