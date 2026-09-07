"""
Regression tests for the audit.

These are not decoration. Three things can silently invalidate the whole
submission, and there is one test class for each:

  1. The vendored GPT-2 vocab is not actually GPT-2  -> every A2/A3 number is junk.
  2. fertility_v2 does not reproduce v0                -> the ablations are not ablations.
  3. A Part B number drifts from the CSV               -> the memo cites a stale figure.

Run: pytest -q  (from partA/, or from repo root)
"""
from __future__ import annotations

import csv
import pathlib
import sys
import unicodedata

import pytest

PARTA = pathlib.Path(__file__).resolve().parents[1]
ROOT = PARTA.parent
sys.path.insert(0, str(PARTA))
sys.path.insert(0, str(ROOT / "partB"))

import fertility_v2 as F           # noqa: E402
import tokenizers_local as T       # noqa: E402


# ---------------------------------------------------------------- 1. tokenizer
class TestTokenizerIntegrity:
    def test_vendored_files_hash(self):
        T._verify_vendor()  # raises on mismatch

    def test_gpt2_known_token_ids(self):
        enc = T.load("gpt2")
        # Public, stable GPT-2 facts. If these drift, the vocab is not GPT-2.
        assert enc("hello world") == [31373, 995]
        assert enc("Hello world") == [15496, 995]
        assert enc(" the") == [262]

    def test_bytes_tokenizer_is_utf8_length(self):
        enc = T.load("bytes")
        for s in ["hello", "नमस्ते", "வணக்கம்", "ಸ್ವಾಗತ"]:
            assert len(enc(s)) == len(s.encode("utf-8"))


# ---------------------------------------------------------------- 2. metrics
class TestMetricCorrectness:
    def test_split_handles_repeated_whitespace(self):
        """The E1 bug: split(' ') invents empty words, split() does not."""
        s = "Please keep the books  in the cupboard."
        assert len(s.split(" ")) == 8
        assert len(s.split()) == 7

    def test_lowercase_is_a_noop_on_devanagari(self):
        """The E2 mechanism: casefolding only touches one arm of the comparison."""
        hin = "मुझे सुबह की चाय बहुत पसंद है।"
        assert hin.lower() == hin
        eng = "The train arrived exactly on time."
        assert eng.lower() != eng

    def test_micro_and_macro_average_differ(self):
        """The E3 bug: mean-of-ratios != ratio-of-sums when line lengths vary."""
        pairs = [(2, 1), (3, 30)]  # (tokens, words)
        macro = sum(t / w for t, w in pairs) / len(pairs)
        micro = sum(t for t, _ in pairs) / sum(w for _, w in pairs)
        assert macro == pytest.approx(1.05)
        assert micro == pytest.approx(5 / 31)
        assert macro != pytest.approx(micro)

    def test_nfc_is_idempotent_and_unifies_nukta(self):
        """The E6 finding: NFC is a real safeguard, not a bug."""
        composed, decomposed = "\u095e", "\u092b\u093c"
        assert composed != decomposed
        n = unicodedata.normalize("NFC", composed)
        assert n == unicodedata.normalize("NFC", decomposed)
        assert n == unicodedata.normalize("NFC", n)

    def test_measure_counts_are_self_consistent(self):
        enc = T.load("gpt2")
        lines = F.read_lines(str(PARTA / "corpora/smoke/eng.txt"))
        m = F.measure("eng", lines, enc)
        assert m.sentences == 12
        # English is pure ASCII here, so codepoints == utf8 bytes == graphemes.
        assert m.codepoints == m.utf8_bytes == m.graphemes
        assert m.tok_per_word == pytest.approx(m.tokens / m.words)

    def test_grapheme_count_never_exceeds_codepoints(self):
        enc = T.load("bytes")
        for lang in ["hin", "kan", "tam", "tel"]:
            m = F.measure(lang, F.read_lines(str(PARTA / f"corpora/smoke/{lang}.txt")), enc)
            assert m.graphemes <= m.codepoints, lang


# ------------------------------------------------- 3. reproduce REPORT_v0
class TestReproducesReportV0:
    """If we cannot reproduce v0 exactly, our 'before' numbers are not v0's."""

    def _v0(self):
        enc = T.load("gpt2")
        out = {}
        for lang, path in [
            ("eng", PARTA / "corpora/sample/eng_sample.txt"),
            ("hin", PARTA / "corpora/sample/hin_sample.txt"),
        ]:
            fert, tpc = [], []
            for raw in open(path, encoding="utf-8"):
                line = raw.strip()
                if not line:
                    continue
                line = unicodedata.normalize("NFC", line).lower()
                t = len(enc(line))
                fert.append(t / len(line.split(" ")))
                tpc.append(t / len(line))
            out[lang] = (sum(fert) / len(fert), sum(tpc) / len(tpc))
        return out

    def test_matches_published_table(self):
        r = self._v0()
        assert round(r["eng"][0], 2) == 1.27
        assert round(r["eng"][1], 3) == 0.226
        assert round(r["hin"][0], 2) == 7.45
        assert round(r["hin"][1], 3) == 1.579
        assert round(r["hin"][0] / r["eng"][0], 2) == 5.89

    def test_corrected_ratio_is_larger_than_reported(self):
        """v0's bugs biased the gap DOWNWARD; the corrected number must exceed 5.89."""
        enc = T.load("gpt2")
        vals = {}
        for lang, path in [
            ("eng", PARTA / "corpora/sample/eng_sample.txt"),
            ("hin", PARTA / "corpora/sample/hin_sample.txt"),
        ]:
            m = F.measure(lang, F.read_lines(str(path)), enc)
            vals[lang] = m.tok_per_word
        corrected = vals["hin"] / vals["eng"]
        assert corrected == pytest.approx(6.11, abs=0.01)
        assert corrected > 5.89


# ---------------------------------------------------------------- 4. Part B
class TestPartB:
    @pytest.fixture(scope="class")
    @classmethod
    def rows(cls):
        with open(ROOT / "partB/bench/bench_log.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_kv_bytes_per_token(self):
        import capacity

        assert 2 * capacity.LAYERS * capacity.KV_HEADS * capacity.HEAD_DIM * 2 == 114_688

    def test_reported_tok_s_is_prompt_plus_gen(self, rows):
        """The B3 claim, checked on every single row."""
        import capacity

        for r in rows:
            assert capacity.total_tp(r) == pytest.approx(float(r["reported_tok_s"]), abs=1.5)

    def test_reported_is_never_goodput(self, rows):
        import capacity

        for r in rows:
            assert capacity.goodput(r) < float(r["reported_tok_s"])

    def test_inflation_factor_is_exactly_structural(self, rows):
        import capacity

        for r in rows:
            p, g = int(r["prompt_len"]), int(r["gen_len"])
            assert float(r["reported_tok_s"]) / capacity.goodput(r) == pytest.approx(
                (p + g) / g, abs=0.01
            )

    def test_batch24_goodput_two_ways_agree(self, rows):
        import capacity

        r = next(r for r in rows if r["batch_size"] == "24")
        way1 = capacity.goodput(r)
        way2 = float(r["reported_tok_s"]) * 512 / 4096
        assert way1 == pytest.approx(200.9, abs=0.1)
        assert way1 == pytest.approx(way2, abs=0.1)

    def test_resident_sequence_count_is_25_from_both_preempting_rows(self, rows):
        """The B1 cross-check. Two independent rows must agree, or B2 is wrong."""
        resident = {
            int(r["batch_size"]) - int(r["preempted_seqs"])
            for r in rows
            if int(r["preempted_seqs"]) > 0
        }
        assert resident == {25}

    def test_goodput_peaks_at_batch_24_not_48(self, rows):
        import capacity

        lc = [r for r in rows if r["prompt_len"] == "3584"]
        best = max(lc, key=capacity.goodput)
        assert best["batch_size"] == "24"

    def test_long_prompts_have_worse_goodput_at_matched_batch(self, rows):
        import capacity

        s = next(r for r in rows if r["batch_size"] == "16" and r["prompt_len"] == "512")
        l = next(r for r in rows if r["batch_size"] == "16" and r["prompt_len"] == "3584")
        assert capacity.goodput(l) < capacity.goodput(s)
