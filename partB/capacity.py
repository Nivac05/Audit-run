#!/usr/bin/env python3
"""
capacity.py -- every number quoted in partB/answers.md, derived from
bench/model_spec.md and bench/bench_log.csv only.

Run:  python capacity.py
Writes results/partB_output.txt
"""
from __future__ import annotations

import csv
import io
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / "bench/bench_log.csv"

# ---- straight from bench/model_spec.md ------------------------------------
LAYERS = 28
KV_HEADS = 8          # GQA: KV heads, NOT the 24 query heads. Using 24 here is
HEAD_DIM = 128        # the single easiest way to get B1 wrong (3x too large).
KV_BYTES = 2          # fp16
PARAMS = 4.2e9
WEIGHT_BYTES = 2      # fp16
GPU_UTIL = 0.92
OVERHEAD_GIB = 1.6
MAX_MODEL_LEN = 4096
GIB = 2 ** 30


def rows():
    with open(LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(r, k):
    v = r[k]
    return float(v) if "." in v else int(v)


def goodput(r):
    """Output (generated) tokens per second -- what the user actually consumes."""
    return num(r, "gen_len") * num(r, "num_requests") / num(r, "wall_clock_s")


def total_tp(r):
    """(prompt + generated) tokens per second -- what reported_tok_s actually is."""
    return (num(r, "prompt_len") + num(r, "gen_len")) * num(r, "num_requests") / num(
        r, "wall_clock_s"
    )


def main(out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    R = rows()

    # ================================================================== B1
    p("=" * 90)
    p("B1 (a) KV-CACHE BYTES PER TOKEN")
    p("=" * 90)
    per_tok = 2 * LAYERS * KV_HEADS * HEAD_DIM * KV_BYTES
    p("  bytes/token = 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes")
    p(f"              = 2 x {LAYERS} x {KV_HEADS} x {HEAD_DIM} x {KV_BYTES}")
    p(f"              = {per_tok:,} bytes = {per_tok/1024:.0f} KiB per token")
    p("  NOTE: kv_heads = 8, not the 24 query heads. This model uses GQA; using 24")
    p("        would overstate KV by exactly 3x and is the classic error here.")

    per_seq = per_tok * MAX_MODEL_LEN
    p(f"\n  per 4096-token sequence = {per_tok:,} x {MAX_MODEL_LEN} = {per_seq:,} B"
      f" = {per_seq/GIB:.4f} GiB")

    p("")
    p("=" * 90)
    p("B1 (b) MAX CONCURRENT 4096-TOKEN SEQUENCES")
    p("=" * 90)
    weights = PARAMS * WEIGHT_BYTES
    p(f"  weights      = {PARAMS:.1e} params x {WEIGHT_BYTES} B = {weights:,.0f} B"
      f" = {weights/GIB:.2f} GiB")
    p(f"  overhead     = {OVERHEAD_GIB} GiB (given)")
    p(f"  gpu_memory_utilization = {GPU_UTIL}")
    p("")
    scenarios = [
        ("nameplate 24 GiB", 24 * GIB),
        ("L4 usable 23034 MiB (what nvidia-smi reports)", 23034 * 2 ** 20),
    ]
    for label, total in scenarios:
        budget = GPU_UTIL * total
        kv = budget - weights - OVERHEAD_GIB * GIB
        p(f"  [{label}]")
        p(f"     usable  = {GPU_UTIL} x {total/GIB:.2f} GiB = {budget/GIB:.2f} GiB")
        p(f"     KV pool = {budget/GIB:.2f} - {weights/GIB:.2f} - {OVERHEAD_GIB:.2f}"
          f" = {kv/GIB:.2f} GiB")
        p(f"     tokens  = {kv/per_tok:,.0f}")
        p(f"     seqs    = {kv/per_seq:.2f}  ->  {int(kv//per_seq)} concurrent 4096-tok sequences")
    p("")
    p("  PREDICTION: 25-28 concurrent sequences, best estimate 25 using real L4 usable memory.")

    p("")
    p("=" * 90)
    p("B1 (c) CHECK AGAINST THE LOG -- three independent derivations")
    p("=" * 90)
    p("  Long-context rows have prompt 3584 + gen 512 = 4096 = exactly max_model_len,")
    p("  so each request occupies one full-length sequence slot. That makes the log")
    p("  directly comparable to the arithmetic above.")
    for r in R:
        pe = num(r, "preempted_seqs")
        if pe:
            bs = num(r, "batch_size")
            p(f"    batch {bs:>2}: {bs} admitted - {pe} preempted = {bs-pe} RESIDENT"
              f"  (kv_cache_util {num(r,'kv_cache_util'):.2f})")
    r24 = next(r for r in R if r["batch_size"] == "24")
    u = num(r24, "kv_cache_util")
    p(f"    batch 24: 24 seqs fit at kv_cache_util {u:.2f} with 0 preemptions")
    p(f"              -> implied capacity {24/u:.2f} sequences")
    p("")
    p("  Batch 32 and batch 48 independently converge on 25 resident sequences, and the")
    p("  batch-24 utilisation extrapolates to 25.8. Predicted 25.76 -> MEASURED 25.")
    p("  The nameplate-24 GiB figure (28) is ~12% high because an L4 exposes ~22.5 GiB,")
    p("  not 24, and vLLM allocates KV in whole blocks.")

    # ================================================================== B3
    p("")
    p("=" * 90)
    p("B3 THE MISREAD COLUMN: reported_tok_s")
    p("=" * 90)
    p("  Hypothesis: reported_tok_s counts PROMPT + GENERATED tokens, i.e. all tokens the")
    p("  GPU touched, not the tokens the user received. Test on all 13 rows:")
    p(f"  {'bs':>3}{'plen':>6}{'glen':>5}{'reported':>10}{'(p+g)n/wall':>13}"
      f"{'gn/wall':>10}{'ratio':>7}")
    ok = True
    for r in R:
        t, g = total_tp(r), goodput(r)
        ok &= abs(t - num(r, "reported_tok_s")) < 1.5
        p(f"  {num(r,'batch_size'):>3}{num(r,'prompt_len'):>6}{num(r,'gen_len'):>5}"
          f"{num(r,'reported_tok_s'):>10.1f}{t:>13.1f}{g:>10.1f}"
          f"{num(r,'reported_tok_s')/g:>7.2f}")
    p(f"\n  All rows match (p+g)*n/wall: {ok}")
    p("  The inflation factor is (prompt+gen)/gen -- a CONSTANT PER PROMPT LENGTH:")
    p("      short rows 512+256 -> 768/256 = 3.00x")
    p("      long  rows 3584+512 -> 4096/512 = 8.00x")
    p("  So the long-prompt rows are inflated 2.67x MORE than the short ones. The report")
    p("  compared an 8x-inflated number (1311.4) against a 3x-inflated number (883.2) and")
    p("  read the difference as a hardware property. Both of its Section 2 conclusions")
    p("  come from this one column.")

    p("")
    p("  HONEST GOODPUT OF THE BATCH-24 LONG-PROMPT ROW -- two independent derivations:")
    g24 = goodput(r24)
    p(f"    (1) from wall clock: gen_len x num_requests / wall_clock_s")
    p(f"        = 512 x 24 / {num(r24,'wall_clock_s')} = {512*24}/{num(r24,'wall_clock_s')}"
      f" = {g24:.1f} output tok/s")
    p(f"    (2) from reported_tok_s: strip the known 8.00x prefill inflation")
    p(f"        = {num(r24,'reported_tok_s')} x (512 / 4096) ="
      f" {num(r24,'reported_tok_s')}/8 = {num(r24,'reported_tok_s')/8:.1f} output tok/s")
    itl = num(r24, "itl_ms_p50")
    p(f"    (3) sanity check from itl_ms_p50 = {itl} ms (steady-state decode ceiling)")
    p(f"        1000/{itl} = {1000/itl:.2f} tok/s per sequence x 25 resident"
      f" = {25*1000/itl:.1f} tok/s")
    p(f"        Higher than {g24:.1f} because wall clock also pays prefill and drain;")
    p(f"        it bounds goodput from above, and {g24:.1f} < {25*1000/itl:.1f}. Consistent.")

    s16 = next(r for r in R if r["batch_size"] == "16" and r["prompt_len"] == "512")
    l16 = next(r for r in R if r["batch_size"] == "16" and r["prompt_len"] == "3584")
    p("")
    p("  'LONGER PROMPTS GIVE BETTER THROUGHPUT' -- at matched batch 16:")
    p(f"    short prompts: reported {num(s16,'reported_tok_s'):.1f} ->"
      f" goodput {goodput(s16):.1f} output tok/s")
    p(f"    long  prompts: reported {num(l16,'reported_tok_s'):.1f} ->"
      f" goodput {goodput(l16):.1f} output tok/s")
    p(f"    Long prompts are {100*(1-goodput(l16)/goodput(s16)):.0f}% WORSE, not better."
      f" The conclusion is inverted.")

    r48 = next(r for r in R if r["batch_size"] == "48")
    best = max(num(r, "reported_tok_s") for r in R)
    p("")
    p("  'BATCH 48 SHOULD GIVE ~3200 tok/s':")
    p(f"    measured at batch 48: reported {num(r48,'reported_tok_s'):.1f},"
      f" goodput {goodput(r48):.1f}")
    p(f"    the projection overstates even the inflated counter by"
      f" {3200/num(r48,'reported_tok_s'):.2f}x, and real goodput by"
      f" {3200/goodput(r48):.1f}x")
    p(f"    (it also mis-sources its own anchor: best reported_tok_s in the log is"
      f" {best:.1f} at batch 64 short, not '~1600')")

    # ================================================================== B2
    p("")
    p("=" * 90)
    p("B2 THE THROUGHPUT ANOMALY IN THE 3584 SWEEP")
    p("=" * 90)
    lc = [r for r in R if r["prompt_len"] == "3584"]
    base = lc[0]
    bg, bb = goodput(base), num(base, "batch_size")
    p(f"  {'bs':>3}{'goodput':>10}{'linear ideal':>14}{'eff':>7}{'wall_s':>9}"
      f"{'ttft_ms':>9}{'itl_ms':>8}{'preempt':>9}{'kv':>6}")
    for r in lc:
        bs = num(r, "batch_size")
        g = goodput(r)
        ideal = bg * bs / bb
        p(f"  {bs:>3}{g:>10.1f}{ideal:>14.1f}{100*g/ideal:>6.0f}%"
          f"{num(r,'wall_clock_s'):>9.2f}{num(r,'ttft_ms_p50'):>9.1f}"
          f"{num(r,'itl_ms_p50'):>8.2f}{num(r,'preempted_seqs'):>9}"
          f"{num(r,'kv_cache_util'):>6.2f}")
    p("")
    p("  ANOMALY: goodput rises monotonically to batch 24 (200.9 tok/s) and then FALLS --")
    p("  173.0 at batch 32 and 162.3 at batch 48. Adding 100% more concurrency makes the")
    p("  system 19% SLOWER in absolute terms. Naive 'throughput scales with batch' predicts")
    p("  424 -> 848 tok/s; we observe 201 -> 162.")
    p("")
    p("  MECHANISM (specific rows and columns):")
    p("   - kv_cache_util goes 0.93 (bs24) -> 0.97 (bs32) -> 0.97 (bs48): the KV pool")
    p("     saturates. B1 says it holds ~25 full-length sequences; 32 and 48 do not fit.")
    p("   - preempted_seqs goes 0 -> 7 -> 23, and 32-7 = 48-23 = 25 resident in both cases.")
    p("     The scheduler admits more than it can hold and evicts the excess.")
    p("   - A preempted sequence loses its KV and must re-prefill 3584 tokens when it is")
    p("     rescheduled. That prefill work is done twice or more and produces no extra")
    p("     output tokens -- pure waste. It shows up as ttft_ms_p50 483->500->637->955")
    p("     (queueing + repeated prefill) while itl_ms_p50 flattens at ~100 ms: the decode")
    p("     step is not getting slower, the requests are simply not resident.")
    p("   - wall_clock_s confirms: 61.16 -> 94.71 -> 151.41 s. Batch 48 takes 2.48x the")
    p("     wall time of batch 24 to do 2.00x the requests.")
    p("   - This is memory-capacity thrash, not compute saturation. At 200 output tok/s")
    p("     the model is nowhere near the L4's 121 TFLOPS or 300 GB/s roofline.")
    p("")
    p("  PROPOSED CHANGE: set max_num_seqs = 24 (from the default 256).")
    two_waves = 2 * num(r24, "wall_clock_s")
    got = num(r48, "wall_clock_s")
    p(f"    The 48-request workload then runs as two clean waves of 24 instead of one")
    p(f"    thrashing wave of 48.")
    p(f"    PREDICTED wall clock: 2 x {num(r24,'wall_clock_s')} = {two_waves:.2f} s")
    p(f"    MEASURED at batch 48:                  {got:.2f} s")
    p(f"    predicted improvement: {100*(1-two_waves/got):.1f}% lower wall clock")
    p(f"    predicted goodput: 48 x 512 / {two_waves:.2f} = {48*512/two_waves:.1f} tok/s"
      f"  vs {goodput(r48):.1f} measured  (+{100*(48*512/two_waves/goodput(r48)-1):.0f}%)")
    p(f"    predicted preempted_seqs: 0 (from 23)")
    p("    Falsifiable: if goodput does not land near 201 tok/s, preemption was not the")
    p("    dominant cost and the mechanism above is wrong.")

    # ================================================================== B4
    p("")
    p("=" * 90)
    p("B4 THE COUNTER TO PULL")
    p("=" * 90)
    p("  vllm:num_preemptions_total (counter), read alongside vllm:num_requests_running")
    p("  and vllm:gpu_cache_usage_perc.")
    p("  EXPECTED VALUES at batch 48, prompt 3584: num_preemptions_total climbs to at")
    p("  least 23 over the run; num_requests_running sits pinned at 25 while 23 requests")
    p("  wait in num_requests_waiting; gpu_cache_usage_perc holds at ~0.97 and never")
    p("  reaches 1.0 (block granularity). The single most diagnostic number is")
    p("  num_requests_running = 25, because B1 predicted 25 from the model spec alone,")
    p("  so agreement confirms the KV-capacity mechanism rather than, say, a CPU-side")
    p("  scheduling or tokenisation bottleneck -- those would leave running at 48.")


if __name__ == "__main__":
    buf = io.StringIO()
    main(buf)
    text = buf.getvalue()
    print(text)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/partB_output.txt").write_text(text, encoding="utf-8")
