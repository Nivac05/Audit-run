# A1 — The evaluation corpus

## Status, stated plainly

The audit machine has no network access to `huggingface.co` or the FLORES
mirrors (verified: both return HTTP 403). So this repo ships **two** corpora and
they are not interchangeable:

| path | what it is | may I quote numbers from it? |
|---|---|---|
| `corpora/flores/` | FLORES-200 dev, 7 languages — built by `build_corpus.py` | **yes, this is the eval set** |
| `corpora/smoke/` | 12 author-written parallel sentences, 5 languages | **no — pipeline smoke test only** |
| `corpora/sample/` | the intern's original 10+10, vendored unchanged | only for reproducing v0 |

`corpora/flores/` is empty in this submission because it could not be downloaded
here. `make corpus` populates it in one command on any networked machine, and
`make a3-full` then regenerates every A3/A4 number against it. Numbers I could
not measure are marked *pending*, not estimated. See `AI_USAGE.md` for why that
line matters to me.

## Choice: FLORES-200

- **Genuinely parallel.** Every line is the same sentence in every language,
  human-translated from one English source. This is the whole reason for the
  choice: A2 §5 argues the only denominator that holds meaning constant is
  per-sentence, and that denominator is meaningless without true line alignment.
  The intern's corpus failed exactly here (A2 §8).
- **Covers the languages the business actually asked about.** English, Hindi,
  Kannada, Tamil, Telugu, Bengali, Marathi — which is the A1 requirement (4+
  languages incl. English, Hindi, and two Dravidian) *and* the exact Part C list.
- **Public and citable**, so anyone reading the memo can re-run it.

Configuration: `dev` split, 997 sentences × 7 languages. `devtest` (1012
sentences) is held back untouched, so if a tokenizer decision ever gets tuned
against `dev` there is a clean set left.

## Preprocessing

Deliberately minimal, and **identical for every language** — see `preprocess()`
in `build_corpus.py`:

1. `strip()` outer whitespace
2. Unicode NFC
3. nothing else

No lowercasing, no punctuation stripping, no whitespace collapsing. The
justification is A2 §2: any per-language cleaning step reintroduces exactly the
asymmetry that sank v0. Token counts should reflect text as it would actually
arrive from a user, punctuation and casing included, because that is what gets
billed.

The one asymmetry I cannot remove is that NFC is not neutral across scripts — it
decomposes Devanagari nukta letters (composition exclusions) while leaving Tamil
untouched. It is still the right call, because the alternative is that the same
sentence tokenises differently depending on which keyboard typed it (A2 §6). I
flag it rather than hide it.

## What this corpus cannot tell you

FLORES-200 is translated news and Wikipedia-register prose. That is a specific
domain and it is **not the domain of a chat assistant**, which is the product
this analysis is meant to inform. Written news Hindi is heavily Sanskritised;
real users type conversational Hindi, frequently romanised or code-mixed with
English. Romanised Hindi tokenises through the *Latin* half of a vocabulary and
would score dramatically better — so a FLORES-derived fertility number is
plausibly a **pessimistic bound** for actual Hindi chat traffic, and I do not
know by how much. The same argument applies with more force to Kannada and Tamil,
where romanised input is common.

Second, 997 sentences is enough for a stable ratio but tells you nothing about
the tail: code-mixing, transliteration, emoji, names, URLs, code blocks, and
formatting markup all tokenise differently and all appear in production traffic.

Third, and most important: this corpus measures **input** text. The product cost
is dominated by generated output, and output length in tokens is a property of
the model's decoding behaviour in each language, not of the corpus. Nothing here
predicts whether the model is more verbose in Tamil.

Fourth, FLORES translations are professional and register-consistent. Whatever
per-language quality variation exists in the translations shows up in the
fertility numbers as if it were a tokenizer property.

**The honest summary: this corpus can rank tokenizers against each other
reliably, and it cannot give you a trustworthy absolute cost multiplier for
production traffic.** The A4 memo is written to respect that distinction, and the
production monitor it proposes exists precisely to close the gap.

## On `corpora/smoke/`

Twelve everyday sentences I wrote and translated myself into Hindi, Kannada,
Tamil and Telugu. Its only job is to let `run_a3.py`, the tests, and the
tokenizer plumbing execute end-to-end with zero network, so that a reviewer can
verify the *pipeline* offline. It is 12 sentences of one register written by one
person; it is not an eval set and no claim in A4 rests on it. Where it appears in
committed output it is banner-marked.
