# A4 — Recommendation memo: tokenizer cost and Indic routing

**To:** Leadership · **Re:** REPORT_v0 §1 · **Bottom line: do not ship the deck.**

## What the report got wrong

REPORT_v0 recommends budgeting 6× serving cost for Hindi and routing all Indic
traffic to a separate model, on the strength of two metrics that "agree". Three
independent problems:

1. **Three code bugs** in `fertility.py`. Corrected, the headline moves 5.89× →
   **6.11×** on the intern's own corpus — the report is wrong in the *conservative*
   direction, which is why it passed review.
2. **The corroboration is circular.** `tok/word` and `tok/char` share a numerator;
   agreement between them confirms nothing. They also do not agree (6.11× vs
   7.39×), and most of the per-character gap is a UTF-8 artefact — per byte it is
   2.81×.
3. **The stated root cause is inverted.** The report says Hindi has more
   characters per word. Measured: **4.75 for Hindi, 5.74 for English.**

## The real finding

This is a **vocabulary-coverage problem in GPT-2**, not a property of Indic
scripts. Against a raw UTF-8 byte tokenizer, GPT-2 saves 76% of tokens on
English, 40% on Hindi, 1.9% on Kannada, and **0.0% on Tamil and Telugu — where its
output is byte-for-byte identical to no tokenizer at all.** 48% of Hindi tokens
are single raw bytes and none exceeds 3 bytes.

That is fixable by changing the tokenizer. The report's conclusion — that this is
intrinsic and the traffic must be routed away — does not follow from its data.

## Headline numbers

| | REPORT_v0 | corrected, same corpus | on FLORES-200 |
|---|---|---|---|
| Hindi vs English, tok/word | 5.89× | **6.11×** | *pending `make a3-full`* |
| Hindi vs English, tok/byte | — | **2.81×** | *pending* |
| **Hindi vs English, tok/parallel-sentence** | — | *not computable — corpus isn't parallel* | *pending — **this is the number to use*** |

I am not filling those cells in. The eval corpus requires network access this
machine does not have, and the one denominator that answers the cost question
cannot be computed on the intern's corpus at all, because it is not parallel
despite being described as such (only 5 of 10 sentences have a translation
partner). Guessing them is the failure mode this audit exists to catch.

## Recommendation

1. **Hold the routing and capacity decision.** It currently rests on a 6× multiple
   derived from the wrong denominator, on a non-parallel corpus, from a
   tokenizer we may not deploy.
2. **Evaluate tokenizers before routing traffic.** Run the corrected pipeline
   across GPT-2, XLM-R and IndicBERTv2 on FLORES-200 and pick on **tokens per
   parallel sentence**. A tokenizer swap is cheaper than a separate serving
   stack, and the byte-fallback evidence says most of the gap is recoverable
   that way.
3. **If a cost multiplier is needed for planning this week**, use the corrected
   **6.1× per word for Hindi with GPT-2** and label it an upper bound with an
   explicit "±, corpus not parallel, one tokenizer, 10 sentences" caveat. Do not
   extrapolate it to Kannada, Tamil or Telugu — those are worse, not equal.

## Biggest caveat

FLORES-200 is translated news prose. Real assistant traffic is conversational and
frequently **romanised or code-mixed** — and romanised Hindi tokenises through the
Latin half of the vocabulary, so it will score far better than Devanagari. Every
number in this analysis is therefore plausibly a **pessimistic bound** on
production cost, by an amount I cannot estimate offline. Additionally, all of it
measures *input*; cost is dominated by *output* length, which is a decoding
property no corpus analysis touches.

## The one metric I would monitor in production

**p50 and p90 of `output_tokens ÷ output_characters`, segmented by detected
language, on live traffic**, with an alert on week-over-week drift.

Why this one: it is measured on the traffic we actually serve, so it is immune to
the corpus-domain caveat above; it is on the **output** side, where the money
actually goes; and dividing by characters of *our own generated text* makes it
comparable across languages without needing a parallel corpus. If the Hindi
figure lands materially below what FLORES predicted, our cost model was built on
the wrong register and the whole analysis needs redoing on real traffic — which is
precisely the failure this audit could not rule out.

Secondary tripwire: **share of output tokens that decode to a single UTF-8 byte,
by language.** That is a direct read on byte fallback. If it rises above ~10% for
any language we serve, the tokenizer is under-covering that script and the
routing question is live again.
