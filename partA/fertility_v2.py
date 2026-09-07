#!/usr/bin/env python3
"""
fertility_v2.py -- corrected tokenizer cost measurement.

Differences from v0 (each one is measured in audit_experiments.py):

  1. Whitespace splitting uses str.split(), not str.split(" "), so runs of
     whitespace do not create phantom empty "words".
  2. No lowercasing. Casefolding is a no-op for Brahmic scripts but changes
     English token counts, so it silently biases exactly one arm of a
     cross-language comparison.
  3. Micro-average (sum(tokens) / sum(denominator)) instead of the mean of
     per-line ratios. A mean of ratios weights a three-word line the same as a
     thirty-word line and is not the corpus-level quantity anyone bills on.
  4. Four denominators are reported side by side, because the choice of
     denominator IS the finding:
         per whitespace word    - not cross-linguistically constant
         per Unicode codepoint  - confounded by script density
         per UTF-8 byte         - encoding-fair, meaning-unfair
         per parallel sentence  - the only one that holds MEANING constant
  5. NFC normalisation is KEPT. v0 was right about this (see audit E6).

Usage:
    python fertility_v2.py --corpus eng=corpora/smoke/eng.txt \
                           --corpus hin=corpora/smoke/hin.txt \
                           --tokenizer gpt2 --baseline eng
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass, asdict

import regex  # for \X grapheme clusters

import tokenizers_local

GRAPHEME = regex.compile(r"\X")


def read_lines(path: str, nfc: bool = True) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if nfc:
                line = unicodedata.normalize("NFC", line)
            out.append(line)
    return out


@dataclass
class Measurement:
    lang: str
    sentences: int
    tokens: int
    words: int
    codepoints: int
    graphemes: int
    utf8_bytes: int

    @property
    def tok_per_word(self) -> float:
        return self.tokens / self.words

    @property
    def tok_per_codepoint(self) -> float:
        return self.tokens / self.codepoints

    @property
    def tok_per_grapheme(self) -> float:
        return self.tokens / self.graphemes

    @property
    def tok_per_byte(self) -> float:
        return self.tokens / self.utf8_bytes

    @property
    def tok_per_sentence(self) -> float:
        return self.tokens / self.sentences

    def row(self) -> dict:
        d = asdict(self)
        d.update(
            tok_per_word=self.tok_per_word,
            tok_per_codepoint=self.tok_per_codepoint,
            tok_per_grapheme=self.tok_per_grapheme,
            tok_per_byte=self.tok_per_byte,
            tok_per_sentence=self.tok_per_sentence,
        )
        return d


def measure(lang: str, lines: list[str], encode) -> Measurement:
    m = Measurement(lang, 0, 0, 0, 0, 0, 0)
    for line in lines:
        m.sentences += 1
        m.tokens += len(encode(line))
        m.words += len(line.split())
        m.codepoints += len(line)
        m.graphemes += len(GRAPHEME.findall(line))
        m.utf8_bytes += len(line.encode("utf-8"))
    return m


DENOMS = [
    ("tok/word", "tok_per_word"),
    ("tok/cp", "tok_per_codepoint"),
    ("tok/graph", "tok_per_grapheme"),
    ("tok/byte", "tok_per_byte"),
    ("tok/sent", "tok_per_sentence"),
]


def report(results: dict[str, Measurement], baseline: str, parallel: bool) -> str:
    out = []
    hdr = f"{'lang':<6}{'sents':>6}{'toks':>7}" + "".join(f"{n:>11}" for n, _ in DENOMS)
    out.append(hdr)
    out.append("-" * len(hdr))
    for lang, m in results.items():
        row = f"{lang:<6}{m.sentences:>6}{m.tokens:>7}"
        row += "".join(f"{getattr(m, a):>11.3f}" for _, a in DENOMS)
        out.append(row)

    if baseline in results and len(results) > 1:
        out.append("")
        out.append(f"ratios vs {baseline}:")
        hdr2 = f"{'lang':<6}" + "".join(f"{n:>11}" for n, _ in DENOMS)
        out.append(hdr2)
        out.append("-" * len(hdr2))
        base = results[baseline]
        for lang, m in results.items():
            if lang == baseline:
                continue
            row = f"{lang:<6}"
            for _, a in DENOMS:
                row += f"{getattr(m, a) / getattr(base, a):>11.2f}"
            out.append(row)
    if not parallel:
        out.append("")
        out.append("WARNING: --parallel not set; the tok/sent column is MEANINGLESS")
        out.append("         unless the corpora are line-aligned translations.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, metavar="LANG=PATH")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--baseline", default=None, help="language code to take ratios against")
    ap.add_argument("--no-nfc", action="store_true")
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="assert the corpora are line-aligned translations (enables tok/sent)",
    )
    ap.add_argument("--json", default=None, help="also write raw counts here")
    args = ap.parse_args()

    encode = tokenizers_local.load(args.tokenizer)

    results: dict[str, Measurement] = {}
    lengths = set()
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        lines = read_lines(path, nfc=not args.no_nfc)
        lengths.add(len(lines))
        results[lang] = measure(lang, lines, encode)

    if args.parallel and len(lengths) != 1:
        raise SystemExit(
            f"--parallel given but corpora have different line counts: {sorted(lengths)}"
        )

    baseline = args.baseline or next(iter(results))
    print(f"tokenizer: {args.tokenizer}")
    print(report(results, baseline, args.parallel))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"tokenizer": args.tokenizer, "results": {k: v.row() for k, v in results.items()}},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
