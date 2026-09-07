# DEFENSE.md — re-derivation sheet

The defense is 30 minutes, screen-shared, with "re-derive this number" and
"add this flag live" as the expected format. This maps every number in the
submission to the command that produces it and the one-line reason it is true.

**If a claim is not in this table, it is not in the submission.**

---

## Setup (once, ~10 s, no network)

```bash
pip install -r requirements.txt
make all
```

If GPT-2 fails to load, run `./tools/fetch_gpt2_vocab.sh` — the loader
sha256-checks the vendored vocab and refuses to run on a mismatch.

---

## Part A

| number | command | why it's true |
|---|---|---|
| **5.89×** (v0 baseline) | `make repro-v0` | the intern's script, unmodified, real GPT-2 |
| **6.11×** (corrected) | `cd partA && python3 audit_experiments.py` → E4 | split() + no-lower + micro-average |
| **+3.1% / 0.0%** lowercase | E2 | Devanagari is unicameral; `.lower()` changes 0/10 Hindi lines |
| **1 phantom word** each corpus | E1 | double space in eng L7, hin L10 |
| **0 effect** of NFC | E6 | 0/10 lines altered; 459 Hindi tokens either way |
| **2 vs 4 tokens** nukta | E6 | U+095E is a composition exclusion → NFC decomposes it |
| **4.75 vs 5.74** chars/word | E7 | codepoints ÷ whitespace words, per corpus |
| **48.4%** single-byte Hindi tokens | E7 | `decode_single_token_bytes` length histogram |
| **tam/tel identical to bytes** | `python3 run_a3.py` | GPT-2 1129 = bytes 1129; GPT-2 946 = bytes 946 |
| **5 of 10** aligned sentences | E8 + read the corpus | hand alignment, printed in E8 |

### Counterfactuals to expect

- *"What if you keep lowercasing but fix the other two?"* → run
  `pipeline(lower=True, split_fix=True, micro=True)`; ratio lands between 5.91
  and 6.11. The lowercase fix is the one carrying most of the 3.8%.
- *"Isn't NFC a bug, it mutates data?"* → measured zero effect **and** removing it
  lets the same sentence tokenise 2× differently by keyboard provenance. It is
  the planted harmless item; flagging it costs 5 points.
- *"Your corrected number is only 4% different. Does the audit matter?"* → the
  code bugs are the *least* important finding. The report's conclusion fails on
  the denominator, the circular corroboration, the inverted root cause, and the
  non-parallel corpus — none of which move the 5.89 number at all.
- *"Why not just use tok/byte, it's encoding-fair?"* → it hands a structural
  advantage to any language UTF-8 encodes in one byte. Nobody is billed per byte.
  It is the right *diagnostic* for vocab coverage, not the right *cost* metric.
- *"Paste this sentence and tokenise it."* →
  `python3 -c "import tokenizers_local as t; e=t.load('gpt2'); print(len(e(input())))"`

### Known weak points — say these before they're asked

- FLORES-200, XLM-R and IndicBERTv2 **never ran**; `huggingface.co` is 403 here.
  A4 has three cells marked *pending*. `make a3-full` fills them.
- `corpora/smoke/` is 12 sentences I translated by hand, unreviewed by a native
  speaker. Its numbers are banner-marked and quoted nowhere.
- `vendor/spm_multi.model` is trained on the text it is measured on. Contaminated
  by construction, used for one qualitative claim only.

---

## Part B

| number | derivation |
|---|---|
| **114,688 B/token** | `2 × 28 layers × 8 KV heads × 128 head_dim × 2 B`. **8, not 24** — GQA |
| **0.4375 GiB/seq** | `114,688 × 4096` |
| **25 sequences** | `(0.92 × 22.49 − 7.82 − 1.6) GiB ÷ 0.4375` = 25.76 |
| **25, from the log** | `32 − 7 = 25` and `48 − 23 = 25`; also `24 ÷ 0.93 = 25.8` |
| **reported_tok_s formula** | `(prompt+gen) × n / wall` — matches all 13 rows |
| **3.00× / 8.00×** inflation | `768/256` and `4096/512` |
| **200.9 tok/s** | `512 × 24 / 61.16` **and** `1607.4 / 8` |
| **260.2 tok/s** ceiling | `(1000/96.07) × 25` — an upper bound, not a third exact route |
| **294.5 vs 163.9** | goodput at batch 16, short vs long |
| **122.32 s** predicted | `2 × 61.16`, two waves of 24 under `max_num_seqs=24` |

`cd partB && python3 capacity.py` prints all of it. Every figure is also asserted
in `partA/tests/test_audit.py::TestPartB`, so `make test` proves none has drifted.

### Counterfactuals to expect

- *"Why 8 KV heads and not 24?"* → GQA. The cache stores K and V, projected to
  the KV head count. Using 24 overstates by exactly 3× and gives 344,064 B.
- *"Your B1 said 28, the log says 25. Which is wrong?"* → neither. 28 assumes a
  nameplate 24 GiB; an L4 exposes ~23,034 MiB. With real usable memory it is
  25.76 → 25, matching two independent log rows. I report both because the
  assumption changes the answer by 12%.
- *"Why is ITL-derived goodput 260 but wall-clock 201?"* → p50 ITL describes
  steady-state decode only; wall clock also pays prefill and drain. 260 bounds
  201 from above. Consistent, not contradictory.
- *"If you set max_num_seqs=24, what breaks?"* → p95 end-to-end gets *worse* for
  the queued half — wave-2 requests wait ~61 s. Throughput and p50 TTFT improve.
  If the p95 SLO dominates, the right answer is a second replica, not a
  scheduler flag.
- *"What if prefix caching was already on during the benchmark?"* → then these
  numbers are optimistic, because all `num_requests` are **identical** and share
  one 3584-token prefix. That is a benchmark-validity problem in its own right.
- *"How would you falsify your B2 mechanism?"* → `num_requests_running` should
  pin at 25 with 48 submitted. If it reads 48 with low cache usage, the
  bottleneck is elsewhere and I'm wrong.

---

## Part C

No numbers to re-derive; be ready to defend the reasoning.

- **Binding constraint is the reviewer, not the A100.** 400 judgements/week × 3
  weeks = 1,200, covering 2 of 6 languages. An SFT set needs 12–30k pairs. Compute
  is ~12 GPU-hours; validated data is the bottleneck by two orders of magnitude.
- **Why not the rewriter?** Costs ~20% of the KV pool (see Part B: 112 KiB/token,
  ~25 sequences), doubles decode latency, and needs the *same* data as SFT.
  Worst cost/benefit.
- **Why 70%?** At n=200, ~5.7σ above chance, ≈5 reviewer-hours per language.
- **Why day 5 for the kill?** Last point at which the SFT fallback still fits
  before a day-16 freeze.
- *"Would you ship to Tamil with no reviewer?"* → only if day 1 shows the effect
  transfers from Hindi to Kannada. Otherwise hold. Shipping an unvalidated style
  change to a language nobody can read is an uncontrolled experiment on users.
