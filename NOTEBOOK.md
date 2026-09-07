# NOTEBOOK

Chronological. Dead ends included, and there were several — two of them changed
conclusions I had already half-written.

---

## Session 1 — read everything before touching anything

Read `REPORT_v0.md`, `fertility.py`, `model_spec.md`, `bench_log.csv`, both
corpora. Wrote down suspicions before running anything, so I could later tell
which ones survived contact with measurement:

1. `line.split(" ")` — looks wrong, both corpora visibly contain a double space
2. `line.lower()` — "so casing doesn't add noise" is suspicious for Devanagari
3. `sum(per_line)/n` — mean of ratios, not a corpus-level number
4. `random.seed(1337)` — why seed a script with no sampling?
5. `unicodedata.normalize("NFC", ...)` — mutating data inside a measurement
6. `tok/char` being used to "confirm" `tok/word` — these share a numerator

Prediction recorded up front: **#5 NFC is the planted harmless one** — it is the
most bug-shaped and the least likely to be an actual bug, and it is exactly what
an AI skim flags first. (This turned out right, but for a reason I had wrong; see
Session 4.)

**Hypothesis:** all of these are measurable one at a time as ablations against a
bit-exact reproduction of v0.

---

## Session 2 — DEAD END, then a recovery: getting the real GPT-2 tokenizer

Everything depends on reproducing REPORT_v0's table exactly. Without that, my
"before" numbers are not the report's numbers and no delta means anything.

- `tiktoken.get_encoding("gpt2")` → **HTTP 403** on
  `openaipublic.blob.core.windows.net`. Sandbox network is allowlisted.
- Checked `huggingface.co` → **403**. So no `transformers`, no XLM-R, no
  IndicBERT, and no FLORES-200. This kills the intended A1/A3 path.
- **Dead end:** looked for a PyPI package bundling a GPT-2 vocab. There isn't
  one; `tiktoken`, `transformers` and `tokenizers` all fetch at runtime.
- **Recovery:** `raw.githubusercontent.com` *is* allowlisted. Found
  `graykode/gpt-2-Pytorch` vendoring `encoder.json` and `vocab.bpe`. Also tried
  `merges.txt` in the same repo (404) and an `ai-forever/gpt2` mirror (404).
- Read `tiktoken.load.read_file` — it takes local paths when the string has no
  `://`, and `read_file_cached` keys the cache on `sha1(url)`. So seeding
  `TIKTOKEN_CACHE_DIR` with two files named after those hashes makes
  `get_encoding("gpt2")` work fully offline.

**Verification, because "a GPT-2-shaped tokenizer" is not good enough:**
`encoder.json` has 50,257 entries, `vocab.bpe` has 50,000 merges, and
`encode("hello world") == [31373, 995]`, `encode("Hello world") == [15496, 995]`,
`encode(" the") == [262]` — public, stable GPT-2 facts. Pinned both sha256s in
`tokenizers_local.py` so a swapped file fails loudly instead of silently
producing plausible garbage.

**Result:** ran v0 unmodified → `eng 1.27 / 0.226, hin 7.45 / 1.579, 5.89×`.
**Bit-exact match to REPORT_v0.** Everything downstream is now a real delta.

*(Minor time sink, logged for honesty: `mkdir -p a/{b,c}` silently created a
directory literally named `{partA` because the shell here is dash, not bash.
Noticed only when `ls` looked wrong.)*

---

## Session 3 — the ablations (A2)

Built `audit_experiments.py`: one parameterised re-implementation of v0's
pipeline, one knob per suspicion, all defaulting to v0 behaviour.

| change | eng | hin | ratio | Δ vs 5.89 |
|---|---|---|---|---|
| v0 baseline | 1.265 | 7.448 | 5.89× | — |
| `split()` | 1.283 | 7.598 | 5.92× | +0.6% |
| no lowercase | 1.229 | 7.448 | 6.06× | +2.9% |
| micro-average | 1.253 | 7.403 | 5.91× | +0.3% |
| **all three** | 1.231 | 7.525 | **6.11×** | **+3.8%** |

**Surprise #1:** every bug pushed the same way — v0 *understates* the gap. I had
assumed at least one would cut the other way. It explains why the report survived
review: it is wrong in the direction that looks conservative.

The lowercasing result is the cleanest thing in the audit: **eng +3.1% tokens,
hin exactly 0.0%, 0/10 Hindi lines even changed.** Devanagari is unicameral, so a
"reduce noise" step is a one-sided bias by construction.

**Surprise #2 (a real dead end):** I had drafted the claim that the three code
bugs were the main story. They are not — 3.8% is a rounding error next to the
conceptual problems. Rewrote the audit to lead with the denominator argument and
demote the code bugs to a table.

---

## Session 4 — NFC: hypothesis wrong, conclusion right

I expected NFC to *matter* here. The Hindi corpus contains `हफ़्ते` and `दफ़्तर`,
and Devanagari nukta letters U+0958–U+095F are Unicode composition exclusions, so
NFC **decomposes** them. I predicted a measurable token delta and a nice finding.

**Measured: 0/10 lines altered in either corpus. Hindi token count 459 either
way. Exactly zero effect.** The corpus was already in decomposed form.

So my mechanism was right and my prediction was wrong. Rather than drop it, I
tested the mechanism synthetically: composed `U+095E` → 2 GPT-2 tokens,
decomposed `U+092B U+093C` → 4 tokens, and NFC maps both to the same string. So
NFC is a genuine safeguard against a 2× discrepancy driven purely by which
keyboard typed the text.

**Revision:** NFC is the "looks suspicious, is actually fine" item — and stronger
than that, removing it would *introduce* a bug. `random.seed(1337)` is a second
inert item, but it is dead code rather than a safeguard, so I report it
separately and explicitly do not claim it as a numerical bug.

---

## Session 5 — the corpus is not what the brief says it is

Went to compute tokens-per-parallel-sentence, since that is where the denominator
argument was heading. Started hand-checking alignment and it fell apart
immediately: `eng[1]` is about Bengaluru airport, `hin[1]` is about liking
morning tea.

Full manual alignment: only 5 of 10 sentences have a translation partner at all,
and only index 3 matches at the same index. Counts corroborate — eng 78 words /
448 codepoints vs hin 61 / 290.

**This was not on my list of suspicions.** The assignment brief itself describes
these as "parallel line-by-line", so I had taken it as given. Lesson recorded:
the premise handed to you is also evidence to be checked.

Consequence: no per-sentence denominator is computable on the intern's corpus,
which is *why* A1 exists. Added `--parallel` to `fertility_v2.py` which refuses to
report `tok/sent` unless line counts match, so this cannot be repeated silently.

---

## Session 6 — DEAD END that became the best finding

Wanted to disprove "this is a property of the script, not the tokenizer". Planned
route: run XLM-R and show low fertility on the same text. **Blocked — no network.**

Fell back to a route that needs no second tokenizer. A byte-level BPE with no
merges for a script degenerates to exactly 1 token per UTF-8 byte, so distance
from that floor measures vocabulary coverage directly:

- eng 4.67 bytes/token, 13.5% single-byte tokens, longest token 14 bytes
- hin 1.66 bytes/token, **48.4%** single-byte tokens, **longest token 3 bytes**

I predicted Hindi would sit at ~1.0 (total fallback). It is 0.601 tok/byte —
partial merges exist. Prediction too strong; recorded as such.

Then ran the 5-language smoke corpus and got the result I did not expect:

| | GPT-2 tokens | UTF-8 bytes |
|---|---|---|
| kan | 933 | 951 |
| **tam** | **1129** | **1129** |
| **tel** | **946** | **946** |

For Tamil and Telugu, GPT-2's output is **byte-for-byte identical to a raw UTF-8
byte tokenizer.** Zero learned merges. Total fallback — the prediction that was
too strong for Hindi is exactly right two languages over.

Then checked the report's stated root cause directly: **Hindi 4.75 codepoints per
word, English 5.74.** The report's explanation is not unproven, it is inverted.

---

## Session 7 — Part B

`reported_tok_s`: guessed it was prompt+gen. Tested on all 13 rows against both
candidate formulas — matches `(prompt+gen)×n/wall` on every row to the printed
precision. Inflation is exactly 3.00× on short rows and 8.00× on long ones, so the
report compared an 8×-inflated number to a 3×-inflated one.

B1 arithmetic: 114,688 B/token. First pass used the **24 query heads** and got
344,064 — caught it re-reading the spec, which lists KV heads separately at 8
under GQA. Logged because it is a 3× error and the exact trap the question is
built around.

Capacity came out **28.9** sequences on nameplate 24 GiB. Log says otherwise:
`32 − 7 = 25` and `48 − 23 = 25`, two independent rows agreeing exactly, plus
`24/0.93 = 25.8`. Redid it with the L4's real usable memory (~23,034 MiB, what
`nvidia-smi` reports) → **25.76 → 25**. Both figures reported, because which
assumption you make changes the answer by 12% and the log adjudicates.

**Third derivation that did not reconcile cleanly:** goodput from `itl_ms_p50`
gives 10.41 tok/s × 25 = 260.2, against 200.9 from wall clock. Spent a while
assuming one was wrong. It is not a contradiction — p50 ITL describes
steady-state decode only, while wall clock also pays prefill and drain, so 260 is
an *upper bound* and 200.9 must sit below it. Kept it as a bound rather than
pretending it was a third exact derivation.

---

## Session 8 — write-up, tests, and one deliberate omission

Wrote 19 regression tests covering the three things that could silently
invalidate the submission: vocab integrity, v0 reproduction, and every Part B
figure. All pass.

**Deliberate omission:** the A4 headline table has three cells reading *pending*.
Those require FLORES-200 and the HF tokenizers, which this machine cannot reach.
The temptation to write plausible numbers there was real and I am flagging it
rather than pretending it wasn't — a 6.1× Hindi figure with a made-up
per-sentence companion would look better and be exactly the failure this audit
was written to catch. `make corpus && make a3-full` fills them in on a networked
machine.
