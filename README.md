# The Audit — submission

An audit of `REPORT_v0.md`, its tokenizer script, and its serving conclusions.

**Every claim here ships with an ablation that produced it.** Where a number could
not be measured on this machine, it is marked *pending* rather than estimated —
see [Honesty ledger](#honesty-ledger).

```bash
pip install -r requirements.txt
make all        # tests + A2 audit + A3 + Part B, all offline, ~10 seconds
```

---

## Headline findings

**Part A — the tokenizer report is wrong three times over.**

| | |
|---|---|
| Reproduced REPORT_v0 §1 **bit-exactly** (1.27 / 0.226 / 7.45 / 1.579 / 5.89×) | so every delta below is a delta on the deck's real numbers |
| Three code bugs, all biasing the same way | **5.89× → 6.11×** — the report errs *conservatively*, which is why it passed review |
| "The two metrics agree, so the result is robust" | void — `tok/word` and `tok/char` **share a numerator**. They also disagree (6.11 vs 7.39) |
| "Hindi has more Unicode characters per word" | **inverted.** Hindi 4.75, English 5.74 |
| "A property of the script, not the tokenizer" | **false.** For Tamil and Telugu, GPT-2's output is *byte-for-byte identical* to a raw UTF-8 byte tokenizer — zero learned merges |
| The corpora are "parallel line-by-line" | **they are not.** Only 5 of 10 sentences have a translation partner |
| `unicodedata.normalize("NFC", …)` | **not a bug.** Zero measured effect, and removing it would introduce one |

**Part B — both §2 conclusions come from one misread column.**

`reported_tok_s` counts **prompt + generated** tokens, verified on all 13 rows.
Inflation is 3.00× on short-prompt rows and 8.00× on long ones, so the report
compared an 8×-inflated number against a 3×-inflated one.

| | |
|---|---|
| KV cache | **114,688 B/token** (112 KiB) — using the 24 *query* heads instead of 8 KV heads is a 3× trap |
| Predicted capacity | **25** concurrent 4096-token sequences |
| Log confirms, twice independently | `32 − 7 = 25` and `48 − 23 = 25` resident |
| Batch-24 long-prompt honest goodput | **200.9 output tok/s** (two independent derivations agree) |
| "Longer prompts give better throughput" | inverted — at batch 16, long prompts are **44% worse** |
| "Batch 48 → ~3200 tok/s" | measured **162.3** goodput. Overstated **19.7×** |

---

## Layout

```
NOTEBOOK.md              chronological log — hypotheses, dead ends, revisions
AI_USAGE.md              where AI helped and where it actively misled
DEFENSE.md               every claim -> the command that re-derives it
Makefile                 every target below

partA/
  AUDIT.md               A2 — the audit, claim by claim, with evidence
  A1_CORPUS.md           A1 — corpus choice, preprocessing, and its limits
  A3_ANALYSIS.md         A3 — tokenizers x denominators, and which number to use
  A4_MEMO.md             A4 — the one-page recommendation
  audit_experiments.py   the A2 ablations (E0-E8)
  fertility_v2.py        corrected metric, five denominators
  run_a3.py              the A3 comparison table
  build_corpus.py        FLORES-200 builder  [needs network]
  train_spm.py           offline Indic tokenizer  [contaminated, diagnostic only]
  tokenizers_local.py    tokenizer registry, sha256-pinned GPT-2
  fertility_v0_reference.py   the intern's script, vendored unchanged
  corpora/{sample,smoke,flores}
  results/               committed output of every script
  tests/test_audit.py    19 regression tests
partB/
  answers.md             B1-B4
  capacity.py            every Part B number
partC/memo.md            the decision memo
tools/fetch_gpt2_vocab.sh
```

## Reproducing

| command | needs network | what it does |
|---|---|---|
| `make test` | no | 19 regression tests |
| `make repro-v0` | no | runs the **intern's original script, unmodified** → 5.89× |
| `make audit` | no | the A2 ablations → `partA/results/a2_audit.txt` |
| `make a3` | no | A3 on the smoke corpus (**not** eval numbers) |
| `make partb` | no | B1–B4 → `partB/results/partB_output.txt` |
| `make corpus` | **yes** | downloads FLORES-200 dev, 7 languages |
| `make a3-full` | **yes** | the real A3 table across five tokenizers |

### How GPT-2 runs offline

Reproducing REPORT_v0 requires the *actual* GPT-2 vocabulary, and this machine
cannot reach `openaipublic.blob.core.windows.net` (HTTP 403). `tiktoken`'s cache
is keyed on `sha1(url)`, so `tools/fetch_gpt2_vocab.sh` vendors `encoder.json`
and `vocab.bpe` from a GitHub mirror and `tokenizers_local.py` seeds the cache
from them.

Both files are **sha256-pinned** and the loader asserts
`encode("hello world") == [31373, 995]` on every load. A substituted or corrupted
vocab fails loudly rather than quietly producing plausible numbers.

## Honesty ledger

What is measured here, and what is not:

- **Measured, real, committed:** the bit-exact v0 reproduction; every A2 ablation;
  the byte-fallback analysis; all of Part B.
- **Measured but on the smoke corpus only:** the A3 table. `corpora/smoke/` is 12
  author-written parallel sentences whose sole job is making the pipeline
  executable offline. It is banner-marked in every output file. **Do not quote it.**
- **Not run:** FLORES-200, XLM-R and IndicBERTv2 — `huggingface.co` returns 403
  here. The code is committed and `make a3-full` runs it in one command.
- **Contaminated by construction:** `partA/vendor/spm_multi.model` is trained on
  the text it is measured on. Labelled everywhere; no claim depends on it.
- **Three cells in `A4_MEMO.md` read *pending*.** They need the eval corpus. They
  are the numbers most worth having and are deliberately left empty.
