# Part B — Capacity reconciliation

All arithmetic is executed by `python capacity.py`; committed output in
`results/partB_output.txt`. Every figure below is asserted in
`partA/tests/test_audit.py::TestPartB`, so a drift breaks the build.

---

## B1 — KV cache and concurrency

### (a) KV-cache bytes per token, exactly

```
bytes/token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
            = 2 × 28 × 8 × 128 × 2
            = 114,688 bytes  =  112 KiB per token
```

**`kv_heads = 8`, not the 24 query heads.** The spec lists 24 attention (Q) heads
and 8 KV heads under GQA. The cache stores K and V, which are projected to the
*KV* head count; using 24 overstates the answer by exactly 3×. This is the single
thing in B1 that is easy to get wrong.

Per 4096-token sequence: `114,688 × 4096 = 469,762,048 B = 0.4375 GiB`.

### (b) Approximate max concurrent 4096-token sequences

Weights: `4.2e9 × 2 B = 8.4 GB = 7.82 GiB`.

| assumption | usable (×0.92) | − weights − 1.6 GiB overhead | KV pool | tokens | **sequences** |
|---|---|---|---|---|---|
| nameplate 24 GiB | 22.08 GiB | | 12.66 GiB | 118,497 | 28.9 → **28** |
| L4 actual ≈22.49 GiB | 20.69 GiB | | 11.27 GiB | 105,527 | 25.8 → **25** |

An L4 nominally "24 GB" exposes about 23,034 MiB to CUDA. I give both because
which one you use changes the answer by 12%, and the log adjudicates between
them.

**Prediction: ~25 concurrent full-length sequences.**

### (c) Check against the log

The long-context rows are `prompt 3584 + gen 512 = 4096` = exactly
`max_model_len`, so each request occupies one full-length slot and the log is
directly comparable to the arithmetic. Three independent derivations:

| source | derivation | resident sequences |
|---|---|---|
| batch 32 row | 32 admitted − 7 preempted | **25** |
| batch 48 row | 48 admitted − 23 preempted | **25** |
| batch 24 row | 24 seqs at `kv_cache_util` 0.93, 0 preempted → 24/0.93 | **25.8** |

Predicted 25.76 → measured 25. The nameplate figure (28) is ~12% high; the
realistic-memory figure is correct to the sequence. That two *different* rows
independently land on exactly 25 is what makes this a real check rather than a
coincidence.

---

## B3 — The misread column *(answered before B2, because B2 depends on it)*

### What it is

**`reported_tok_s` counts prompt + generated tokens** — every token the GPU
touched — not the tokens delivered to the user. Verified on **all 13 rows**:

```
reported_tok_s == (prompt_len + gen_len) × num_requests / wall_clock_s
```

Exact to the printed precision on every row (`test_reported_tok_s_is_prompt_plus_gen`).

The inflation over true output throughput is therefore structural and constant
per prompt length:

- short rows: `768 / 256` = **3.00×**
- long rows: `4096 / 512` = **8.00×**

Long-prompt rows are inflated **2.67× more** than short-prompt rows. Both of
REPORT_v0 §2's conclusions fall out of that one fact.

### Honest goodput of the batch-24 long-prompt row

**200.9 output tokens/second.** Two independent derivations:

1. **From wall clock:** `512 × 24 / 61.16 = 12,288 / 61.16 = 200.9 tok/s`
2. **From the reported counter:** strip the known 8.00× prefill inflation —
   `1607.4 × (512/4096) = 1607.4 / 8 = 200.9 tok/s`

Third sanity check, from a different column: `itl_ms_p50 = 96.07 ms` →
`1000/96.07 = 10.41 tok/s` per sequence × 25 resident = **260.2 tok/s** steady-state
decode ceiling. Goodput must sit below it because wall clock also pays prefill
and drain, and 200.9 < 260.2. Consistent.

### What the report should have said

Both erroneous conclusions, corrected:

**"Longer prompts clearly give better GPU utilization"** — inverted. At matched
batch 16:

| | reported_tok_s | **goodput** |
|---|---|---|
| short prompts (512) | 883.2 | **294.5 tok/s** |
| long prompts (3584) | 1311.4 | **163.9 tok/s** |

Long prompts are **44% worse**, not better. The 1311 > 883 comparison is an
8×-inflated number against a 3×-inflated one.

**"Batch 48 should give ~3200 tok/s"** — the log already contains batch 48, and it
gives **1298.5 reported / 162.3 goodput**. The projection overstates even the
inflated counter by 2.46× and real goodput by **19.7×**. It also mis-sources its
own anchor: the best `reported_tok_s` in the log is 2267.3 (batch 64, short
prompts), not "~1600".

> **Honest wording:** "Peak measured output goodput is 294.5 tok/s at batch 16
> with short prompts, and 200.9 tok/s at batch 24 with 3584-token prompts.
> Throughput does **not** scale past batch 24 on long prompts — it declines,
> because the KV cache holds only ~25 full-length sequences. Capacity planning
> should assume ~200 output tok/s per L4 for long-context traffic and must not
> extrapolate linearly in batch size."

---

## B2 — The throughput anomaly in the 3584 sweep

| bs | goodput | linear ideal | efficiency | wall_s | ttft_ms | itl_ms | preempted | kv_util |
|---|---|---|---|---|---|---|---|---|
| 4 | 70.7 | 70.7 | 100% | 28.98 | 483.2 | 51.33 | 0 | 0.16 |
| 8 | 112.8 | 141.3 | 80% | 36.30 | 519.0 | 62.26 | 0 | 0.31 |
| 16 | 163.9 | 282.7 | 58% | 49.97 | 498.3 | 77.20 | 0 | 0.62 |
| **24** | **200.9** | 424.0 | 47% | 61.16 | 500.5 | 96.07 | 0 | 0.93 |
| 32 | 173.0 | 565.4 | 31% | 94.71 | 636.9 | 101.79 | **7** | 0.97 |
| 48 | 162.3 | 848.0 | 19% | 151.41 | 955.4 | 100.00 | **23** | 0.97 |

**The anomaly:** goodput rises to batch 24 and then *falls*. Doubling concurrency
from 24 to 48 makes the system **19% slower in absolute terms** (200.9 → 162.3
tok/s). Naive linear scaling predicts 848 tok/s; we get 162.

### Mechanism, by row and column

- **`kv_cache_util`: 0.93 → 0.97 → 0.97.** The KV pool saturates between batch 24
  and 32. B1 says it holds ~25 full-length sequences; 32 and 48 do not fit.
- **`preempted_seqs`: 0 → 7 → 23.** And `32 − 7 = 48 − 23 = 25` resident in both
  cases. The scheduler admits more sequences than the cache can hold and evicts
  the excess. This is the direct fingerprint.
- **The cost of eviction is repeated prefill.** A preempted sequence loses its KV
  blocks and must re-prefill all 3584 prompt tokens when rescheduled. That work
  produces zero additional output tokens. At 3584 prompt vs 512 gen, prefill is
  87.5% of each request's token volume, so paying it twice is catastrophic — which
  is exactly why the anomaly appears in the long-prompt sweep and not the short
  one (the 512-prompt sweep reaches batch 64 with `kv_util` only 0.47 and zero
  preemptions).
- **`ttft_ms_p50`: 500.5 → 636.9 → 955.4**, while **`itl_ms_p50` flattens at
  ~100 ms.** This distinguishes the mechanism from compute saturation. Decode
  steps are *not* getting slower; requests are simply not resident and are
  queueing behind re-prefill. If this were compute-bound, ITL would climb with
  batch and TTFT would not spike disproportionately.
- **`wall_clock_s`: 61.16 → 94.71 → 151.41.** Batch 48 takes **2.48×** the wall time
  of batch 24 to serve **2.00×** the requests.

This is memory-capacity thrash, not compute saturation. At ~200 output tok/s the
model is nowhere near the L4's 121 TFLOPS or 300 GB/s roofline.

### Proposed change, with a quantitative prediction

**Set `max_num_seqs = 24`** (from the vLLM default of 256).

The scheduler then never admits more than the KV cache can hold, so the
48-request workload runs as two clean waves of 24 instead of one thrashing wave
of 48.

| | predicted | measured today |
|---|---|---|
| wall clock for 48 requests | `2 × 61.16 = 122.32 s` | 151.41 s |
| improvement | **−19.2%** | |
| goodput | `48 × 512 / 122.32 = 200.9 tok/s` | 162.3 tok/s (**+24%**) |
| `preempted_seqs` | **0** | 23 |
| `ttft_ms_p50` | ~500 ms for wave 1, ~61 s queue delay for wave 2 | 955.4 ms |

Note the honest trade-off: admission control **improves throughput and p50 TTFT
but worsens p95 end-to-end latency for the queued half**, because wave-2 requests
wait ~61 s. If the p95 SLO matters more than throughput, the correct fix is a
second replica, not a scheduler tweak.

**Falsifiable:** if goodput after the change does not land near 200 tok/s,
preemption was not the dominant cost and this mechanism is wrong.

*Alternative considered and rejected:* enabling prefix caching would collapse
prefill almost entirely here — but only because the harness sends
`num_requests` **identical** requests, so all 48 share one prefix. That is a
property of the benchmark, not of production traffic, and optimising for it would
be tuning to an artefact. Worth flagging as a benchmark-validity problem in its
own right: these numbers may already be optimistic if any prefix caching was on.

---

## B4 — The counter to pull

**`vllm:num_preemptions_total`**, read alongside `vllm:num_requests_running`,
`vllm:num_requests_waiting` and `vllm:gpu_cache_usage_perc`.

Expected values on a re-run of the batch-48 / prompt-3584 row:
`num_preemptions_total` climbs to **at least 23** over the run (more if sequences
are preempted repeatedly); `num_requests_running` sits pinned at **25** while the
remaining 23 sit in `num_requests_waiting`; `gpu_cache_usage_perc` holds at
**~0.97** and never reaches 1.0, because vLLM allocates KV in whole blocks and
cannot use the remainder.

`num_requests_running = 25` is the single most diagnostic number, because B1
predicted 25 from the model spec alone, by a completely independent route
(arithmetic on layers, KV heads and head_dim). If the serving stack reports 25
running while 48 were submitted, the KV-capacity mechanism is confirmed. If it
instead reports 48 running with low cache usage, the bottleneck is somewhere
else — CPU-side scheduling, tokenisation, or the harness — and my B2 explanation
is wrong.
