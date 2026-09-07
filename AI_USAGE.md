# AI_USAGE.md

> **Note to the reader:** this file must be accurate about *this* submission and
> about who understands what. If you are picking this repo up, edit it to match
> your own working process before submitting — a disclosure written by someone
> else is worse than no disclosure.

## Summary

Claude (Opus) was used heavily throughout, in an agentic setup with a shell,
Python, and a network-restricted sandbox. It wrote most of the code and drafted
most of the prose. The measurements are real: every number in this repo was
produced by executing the committed scripts in that sandbox, and the outputs are
committed under `partA/results/` and `partB/results/`.

## Where AI genuinely helped

- **Recovering the GPT-2 tokenizer offline.** The sandbox blocks
  `openaipublic.blob.core.windows.net`. Reading `tiktoken.load` to discover that
  `read_file` accepts local paths and that `read_file_cached` keys on
  `sha1(url)` — and therefore that the cache could be seeded from a GitHub
  mirror — was the step that made bit-exact reproduction of REPORT_v0 possible.
  Without it, no ablation in A2 would have been meaningful.
- **Mechanical ablation scaffolding.** The parameterised re-implementation of
  `analyze()` with one knob per suspected bug is boilerplate, and generating it
  was fast.
- **Part B arithmetic and cross-checks.** Testing `reported_tok_s` against both
  candidate formulas on all 13 rows, and noticing that `32 − 7` and `48 − 23`
  both equal 25.
- **Prose density.** The memos were drafted and then cut down.

## Where AI misled me, or would have

These are the ones worth reading.

- **It flags NFC as a bug on sight.** Asked cold what is wrong with
  `fertility.py`, the confident answer includes "it silently mutates your input
  data with NFC normalisation". That is the planted trap, and it is worth −5.
  Only measuring it (0/10 lines altered, identical token counts) and then testing
  the mechanism synthetically established that it is not merely harmless but
  protective. **The instinct to flag it was exactly wrong.**
- **It accepted the brief's framing that the corpora are parallel.** Both the
  assignment PDF and the initial analysis treated `corpus_sample/` as
  line-aligned. It is not — only 5 of 10 sentences have a translation partner.
  This surfaced only from reading the sentences by hand, and it invalidates more
  of REPORT_v0 §1 than any of the code bugs do.
- **Overconfident magnitude predictions.** Predicted Hindi would sit at ~1.0
  tokens/byte (total byte fallback). Measured 0.601 — partial merges exist. The
  prediction was right in mechanism, wrong in magnitude, and it happened to
  become correct two languages over (Tamil and Telugu are at exactly 1.000). If
  I had written the prediction up without measuring, it would have been a
  confident, plausible, wrong claim about Hindi.
- **The GQA trap.** First pass at B1 used the 24 query heads instead of the 8 KV
  heads — a clean 3× error, and one that produces an entirely plausible-looking
  number. Caught by re-reading the spec, not by anything the model volunteered.
- **Strong pull toward filling in missing numbers.** With FLORES-200 unreachable,
  there was constant pressure to produce a "corrected" cross-language table
  anyway. Those cells are marked *pending* in `A4_MEMO.md` instead. This is the
  single most important line in this file.

## What is measured vs what is not

| | status |
|---|---|
| A2 — every ablation, bit-exact v0 reproduction | **measured**, real GPT-2, committed output |
| A3 — GPT-2 and `bytes` on `corpora/smoke/` | **measured**, but smoke corpus only |
| A3 — XLM-R / IndicBERTv2 on FLORES-200 | **not run** — no network. Code committed, `make a3-full` |
| A1 — FLORES-200 corpus | **not downloaded** — no network. `build_corpus.py` committed |
| B1–B4 | **fully measured** from the provided spec and log, all asserted in tests |
| C | reasoning; the arithmetic is back-of-envelope by construction |

The SentencePiece tokenizer in `partA/vendor/` is trained on the same text it is
measured on. It is contaminated, labelled as such everywhere it appears, and no
claim depends on its numbers.

`corpora/smoke/` is 12 sentences translated by hand into four Indic languages.
Treat the translations as the weakest link in the repo — they are one person's
everyday register, unreviewed by a native speaker, and they exist only to make
the pipeline executable offline. No reported conclusion rests on them.

## Verification

`pytest partA/tests/test_audit.py` — 19 tests, all passing. They cover the three
failure modes that would silently invalidate everything: a wrong vocab, a failure
to reproduce v0, and any Part B number drifting from the CSV.
