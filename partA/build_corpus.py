#!/usr/bin/env python3
"""
build_corpus.py -- assemble the A1 evaluation corpus from FLORES-200.

WHY FLORES-200
  It is genuinely parallel (every line is the same sentence in every language),
  human-translated from a single English source, covers all six languages the
  Part C product ask names, and is a public benchmark so the numbers are
  reproducible by whoever reads the memo. Genuine parallelism is the whole
  point: it is the only property that makes a per-sentence denominator
  meaningful (see A3).

REQUIRES NETWORK. The audit sandbox blocks huggingface.co and the FLORES
mirrors, so this script is the documented path, not something that ran here.
Everything under corpora/smoke/ is a 12-sentence author-written stand-in used
ONLY to prove the pipeline executes end to end; no reported number depends on it.

Usage:
    python build_corpus.py --out corpora/flores --split dev
    python build_corpus.py --out corpora/flores --split dev --via hf
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import unicodedata

# FLORES-200 language codes -> our short codes.
LANGS = {
    "eng_Latn": "eng",
    "hin_Deva": "hin",
    "kan_Knda": "kan",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
    "ben_Beng": "ben",
    "mar_Deva": "mar",
}


def preprocess(line: str) -> str:
    """
    Deliberately minimal, and identical for every language.

    Any per-language cleaning would reintroduce exactly the asymmetry that
    sinks fertility.py v0 (see audit E2). So: strip outer whitespace, NFC, and
    nothing else. In particular we do NOT lowercase, do not strip punctuation
    and do not collapse internal whitespace -- token counts must reflect text
    as it would actually arrive from a user.
    """
    return unicodedata.normalize("NFC", line.strip())


def via_hf(split: str):
    from datasets import load_dataset

    for flores_code, short in LANGS.items():
        ds = load_dataset("facebook/flores", flores_code, split=split)
        yield short, [preprocess(r["sentence"]) for r in ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpora/flores")
    ap.add_argument("--split", default="dev", choices=["dev", "devtest"])
    ap.add_argument("--via", default="hf", choices=["hf"])
    ap.add_argument("--limit", type=int, default=None, help="truncate to N sentences")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        pairs = dict(via_hf(args.split))
    except ImportError as e:
        sys.exit(
            f"missing dependency: {e}\n"
            "  pip install datasets\n"
            "(This is a dependency problem, not a network problem.)"
        )
    except Exception as e:
        sys.exit(
            f"corpus download failed: {type(e).__name__}: {e}\n"
            "Most likely this machine cannot reach huggingface.co (the audit sandbox\n"
            "returns HTTP 403). Run this on a networked machine.\n"
            "corpora/smoke/ is a pipeline smoke test and is NOT a substitute."
        )

    n = {len(v) for v in pairs.values()}
    if len(n) != 1:
        sys.exit(f"FATAL: languages have different line counts {n} -- not parallel, aborting.")

    for short, lines in pairs.items():
        if args.limit:
            lines = lines[: args.limit]
        (out / f"{short}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out/short}.txt  ({len(lines)} sentences)")

    print(f"\nparallel line count verified: {n.pop()} sentences x {len(pairs)} languages")


if __name__ == "__main__":
    main()
