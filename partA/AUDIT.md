# A2 — Audit of `fertility.py` and of the metric

Every number below is produced by `python audit_experiments.py`
(full output committed at `results/a2_audit.txt`). Nothing here is asserted
without an ablation behind it.

## 0. Baseline: we reproduce REPORT_v0 bit-exactly

| | eng fert | eng tok/char | hin fert | hin tok/char | ratio |
|---|---|---|---|---|---|
| REPORT_v0 §1 | 1.27 | 0.226 | 7.45 | 1.579 | 5.89× |
| our re-run | 1.27 | 0.226 | 7.45 | 1.579 | 5.89× |

This matters more than it looks. It means every delta in this document is a
delta on the exact numbers in the leadership deck, not on a lookalike. It also
required getting the real GPT-2 vocab offline — see `tools/fetch_gpt2_vocab.sh`
and the sha256 pins in `tokenizers_local.py`.

---

## Claims table

| # | Thing | Verdict | Effect on the 5.89× headline |
|---|---|---|---|
| 1 | `line.split(" ")` | **bug** | +0.6% (5.89 → 5.92) |
| 2 | `line.lower()` | **bug**, asymmetric | +2.9% (5.89 → 6.06) |
| 3 | mean-of-per-line-ratios | **bug** | +0.3% (5.89 → 5.91) |
| 1+2+3 | all code bugs | | **5.89 → 6.11×** |
| 4 | "the two metrics agree, so it's robust" | **conceptual error** | invalidates the reasoning, not just the number |
| 5 | tokens-per-*word* as the cost metric | **conceptual error** | the metric answers the wrong question entirely |
| 6 | `unicodedata.normalize("NFC", …)` | **NOT a bug — correct and protective** | 0.00% (measured) |
| 7 | "property of the script, not the tokenizer" | **report claim is factually inverted** | — |
| 8 | corpora described as "parallel" | **corpus defect** | confounds every ratio |

---

## 1. `line.split(" ")` — bug (E1)

`split(" ")` on a run of two spaces yields an empty string that is then counted
as a word.

```
eng line 7 : split(' ')→8 words, split()→7   'Please keep the books  in the cupboard.'
hin line 10: split(' ')→6 words, split()→5   'किताबें  अलमारी में रखी हैं।'
```

Phantom words inflate the denominator, so v0 **understates** fertility.
Fixing it alone: 5.89 → 5.92×. Small here only because both corpora happen to
contain exactly one double space, so it partly cancels in the ratio. That is
luck. On a corpus where only one language has messy whitespace it would not
cancel, and the direction of the error would be unpredictable.

## 2. `line.lower()` — the important code bug (E2)

The comment says "lowercase so casing doesn't add noise to the comparison". It
does the opposite, because Devanagari is unicameral:

| | tokens cased | tokens lowered | Δ | lines changed by `.lower()` |
|---|---|---|---|---|
| eng | 96 | 99 | **+3.1%** | 10/10 |
| hin | 459 | 459 | **0.0%** | 0/10 |

A normalisation step that is a no-op on one arm of a two-arm comparison and a
+3.1% penalty on the other is not noise reduction, it is a bias. It inflates the
English baseline, which **shrinks** the measured gap: removing it moves 5.89 →
6.06×.

## 3. Mean of ratios instead of micro-average (E3)

`sum(t_i/w_i)/n` weights a 3-word line identically to a 14-word line. Nobody is
billed per line; the quantity that maps to cost is `sum(tokens)/sum(words)`.
Effect alone: 5.89 → 5.91×. Small on this corpus, unbounded in general —
`test_micro_and_macro_average_differ` shows a constructed case where the two
differ by 6×.

**All three code bugs fixed: 5.89× → 6.11×.** Note the direction: v0's bugs all
made Hindi look *better* than it is. The report errs conservatively, which is
exactly why it survived review.

---

## 4. The conceptual error, part 1: "the two metrics agree"

REPORT_v0 finding 2 says the tok/char column "confirms the per-word number", and
the recommendation leans on it: *"No further measurement needed — the two metrics
agree, so the result is robust."*

This is void for three separate reasons.

**(a) The metrics are not independent.** `tok/word` and `tok/char` share the same
numerator. Any error in the token count — a tokenizer misconfiguration, a
lowercasing bug, an encoding problem — moves both in the same direction by the
same factor. Agreement between them carries *zero* confirmatory information. It
is the same measurement twice.

**(b) They do not actually agree.** Corrected: 6.11× per word vs 7.39× per
codepoint. A 21% spread is waved through as "agreement".

**(c) The per-codepoint gap is mostly an artefact of the denominator.** Devanagari
costs 2.63 UTF-8 bytes per codepoint; ASCII costs 1.00. Normalise that away and
the gap falls to **2.81× per UTF-8 byte**. The "7.0× worse per character" headline
is measuring UTF-8, not the tokenizer.

## 5. The conceptual error, part 2 — the one that matters

> The code computes exactly what it says. What it says is the wrong thing to compute.

`fertility.py` computes **tokens per whitespace word**, and the report converts
that directly into **"Hindi will cost us roughly 6× more per request"**.

That conversion is invalid, because a *word* is not a cross-linguistically
constant unit. Languages package the same meaning into different numbers of
words. On our parallel smoke corpus, holding meaning fixed:

| | words per parallel sentence |
|---|---|
| eng | 5.58 |
| hin | 5.17 |
| kan | 4.00 |
| tam | 4.08 |
| tel | 4.08 |

Kannada uses 28% fewer words than English to say the identical sentence, because
it is agglutinative — morphology that English spreads across several words is
bound into one. So `tok/word` **double-counts agglutination**: it divides a large
token count by a small word count and reports the product of two effects as if it
were one.

What a routing-and-cost decision needs is tokens per unit of *delivered meaning*,
because tokens are what you are billed for and meaning is what the user asked
for. The denominator has to hold **content** constant, not orthography. That
requires a genuinely parallel corpus and a per-sentence denominator. See A3.

## 6. Looks suspicious, is actually correct: NFC normalisation

`unicodedata.normalize("NFC", line)` silently mutates input data inside a
measurement script. That is the shape of a bug, and it is the thing an
AI-assisted skim flags first. It is not a bug.

**Measured effect on the reported numbers: exactly zero.** 0/10 lines are altered
by NFC in either corpus; Hindi token count is 459 either way.

And it is actively protective. Devanagari nukta letters have two legal encodings,
which GPT-2 tokenises differently:

```
composed   U+095E              → 2 tokens
decomposed U+092B U+093C       → 4 tokens
after NFC  both → U+092B U+093C → 4 tokens, identical
```

Without NFC, two visually identical corpora could differ by 2× in token count
purely by encoding provenance. **Removing this line would introduce a real bug.**
I do not claim it as a flaw.

*Secondary, and I am explicitly not claiming it as a numerical bug:*
`random.seed(1337)  # reproducibility` is inert. The script imports `random`,
seeds it, and never samples. Provable zero effect. It is misleading dead code —
the comment implies a sampling step that does not exist, which invites a reader
to assume the corpus was subsampled — and it should be deleted. But deleting it
changes no number.

## 7. Report finding 3 is factually inverted (E7)

> "Hindi simply has more Unicode characters per word, so any tokenizer will
> struggle. This is a property of the script, not the tokenizer."

Measured on the intern's own corpus:

| | codepoints per word |
|---|---|
| eng | **5.74** |
| hin | **4.75** |

Hindi has *fewer* characters per word than English. The stated root cause is not
merely unsupported; it is backwards.

The actual mechanism is vocabulary allocation. A byte-level BPE with no merges
for a script degenerates to exactly 1 token per UTF-8 byte, so distance from that
floor measures coverage:

| | bytes/token | single-byte tokens | longest token |
|---|---|---|---|
| eng | 4.67 | 13.5% | 14 bytes |
| hin | 1.66 | **48.4%** | **3 bytes** |

Nearly half of all Hindi tokens are raw single bytes, and **no Hindi token
exceeds 3 bytes** — GPT-2 never merges beyond a single Devanagari codepoint.
English reaches 14.

It gets starker on Dravidian languages (`run_a3.py`):

| | GPT-2 tokens | raw UTF-8 bytes | verdict |
|---|---|---|---|
| eng | 84 | 354 | GPT-2 saves 76.3% |
| hin | 470 | 782 | saves 39.9% |
| kan | 933 | 951 | saves 1.9% |
| tam | 1129 | 1129 | **identical — total byte fallback** |
| tel | 946 | 946 | **identical — total byte fallback** |

For Tamil and Telugu, GPT-2 is *exactly* a UTF-8 byte tokenizer. Zero learned
merges. This is a property of GPT-2's training mixture, and it is fixable by
choosing a different tokenizer — which directly undermines the report's
"no further measurement needed / budget 6× and route Indic traffic away".

## 8. Corpus defect: the samples are not parallel (E8)

The brief describes `corpus_sample/` as "parallel line-by-line". Hand alignment
shows it is not:

| eng | hin | content |
|---|---|---|
| 3 | 3 | bought this book yesterday (only same-index match) |
| 5 | 6 | train arrived on time |
| 4 | 7 | children playing cricket |
| 8 | 4 | visiting Mysuru next week |
| 7 | 10 | books in the cupboard |
| 1,2,6,9,10 | — | no counterpart |
| — | 1,2,5,8,9 | no counterpart |

Only 5 of 10 sentences have a translation partner at all, and only one sits at a
matching index. Corroborating counts: English 78 words / 448 codepoints against
Hindi 61 / 290 — the English side simply says more.

Consequence: the two corpora do not express the same content, so **every ratio in
REPORT_v0 §1 is confounded by content difference**, on top of the code bugs. This
is unfixable by better statistics; it needs a different corpus, which is A1.

---

## What I checked and found clean

Listed so the defense can probe them, and so it is clear the audit was not just
pattern-matching for bugs:

- `add_special_tokens=False` on the HF path — correct. Including specials would
  add a constant per-sentence offset that biases short sentences.
- `strip()` and skipping blank lines — correct and symmetric.
- Reading with `encoding="utf-8"` explicitly — correct; relying on the locale
  default would be a real portability bug.
- `results[lang]` insertion order for choosing the ratio baseline — deterministic
  in Python 3.7+, matches argument order. Fragile but not wrong.
- No division-by-zero path exists: blank lines are skipped before division, and
  `split(" ")` on a non-empty string always returns ≥1 element.
