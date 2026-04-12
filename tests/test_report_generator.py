# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for ``benchmarks.report_generator``.

Covers:

* ``load_json`` happy path, missing file, malformed file
* ``find_judge_results`` schema sniffing across v2 ``aggrefact_eval``
  layout, the routed Gemma layout, and explicit rejection of unrelated
  JSONs
* ``format_judge_table`` content + empty case
* ``format_rust_table`` from a real-shape ``rust_compute_bench.json``
  payload + empty case
* ``format_external_leaderboard`` from a hand-curated entries list +
  empty case
* ``format_oracle_section`` labels Oracle as theoretical upper bound,
  not as a leaderboard match
* ``generate_report`` end-to-end: writes file, no banned words, no
  fabricated $/USD currency, no hardcoded "v4.0.0-rc1" string, every
  judge JSON makes it into the output, missing inputs degrade
  gracefully (no crash, no fabrication)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from report_generator import (  # noqa: E402
    find_judge_results,
    format_external_leaderboard,
    format_judge_table,
    format_oracle_section,
    format_rust_table,
    generate_report,
    load_json,
)

# ── Fixture builders ─────────────────────────────────────────────────────


def _aggrefact_eval_v2(
    label: str = "yaxili96/FactCG-DeBERTa-v3-Large",
    global_ba: float = 0.7558,
    per_ds_avg: float = 0.7776,
) -> dict:
    return {
        "schema_version": 2,
        "model": label,
        "backend": "transformers",
        "samples": 29320,
        "global_balanced_accuracy": global_ba,
        "global_threshold": 0.46,
        "per_dataset_avg_balanced_accuracy": per_ds_avg,
        "per_dataset": {"AggreFact-CNN": {"balanced_acc": 0.74}},
        "per_dataset_thresholds": {"AggreFact-CNN": 0.5},
        "scores": [0.5, 0.6],
        "predictions": [0, 1],
        "labels": [0, 1],
        "datasets_per_sample": ["a", "b"],
    }


def _routed_gemma_v2(
    label: str = "/tmp/gemma-models/gemma-4-E4B-Q6.gguf",
    global_ba: float = 0.8211,
) -> dict:
    return {
        "model": label,
        "method": "per-dataset prompt routing (summ/rag/claim families)",
        "samples": 29320,
        "global_balanced_accuracy": global_ba,
        "per_dataset": {"AggreFact-CNN": {"balanced_acc": 0.78}},
        "per_family": {"summ": 0.78, "rag": 0.79, "claim": 0.87},
        "predictions": [1, 0, 1],
        "labels": [1, 0, 1],
        "datasets_per_sample": ["x", "y", "z"],
        "families_per_sample": ["summ", "rag", "claim"],
    }


def _rust_compute_bench() -> dict:
    return {
        "benchmark": "rust_compute",
        "iterations": 5000,
        "rust_available": True,
        "results": {
            "sanitizer_score": {
                "description": "11 injection regex patterns",
                "python": {"median_us": 57.59},
                "rust": {"median_us": 2.13},
                "speedup": 27.04,
            },
            "temporal_freshness": {
                "description": "datetime parse",
                "python": {"median_us": 53.0},
                "rust": {"median_us": 2.5},
                "speedup": 21.2,
            },
        },
    }


def _external_leaderboard() -> dict:
    return {
        "entries": [
            {
                "system": "FaithLens 8B",
                "params": "8B",
                "balanced_accuracy": 0.864,
                "source_url": "https://example.org/faithlens",
            },
            {
                "system": "Bespoke-MiniCheck-7B",
                "params": "7B",
                "balanced_accuracy": 0.778,
                "source_url": "https://example.org/minicheck",
            },
        ]
    }


def _sentinel_payload() -> dict:
    return {
        "voting_balanced_accuracy": 0.81,
        "routed_balanced_accuracy": 0.83,
        "fusion_balanced_accuracy": 0.84,
        "oracle_balanced_accuracy": 0.86,
    }


# ── load_json ────────────────────────────────────────────────────────────


class TestLoadJson:
    def test_happy_path(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text('{"a": 1}')
        assert load_json(p) == {"a": 1}

    def test_missing_file_returns_none(self, tmp_path):
        assert load_json(tmp_path / "missing.json") is None

    def test_malformed_file_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not: valid")
        assert load_json(p) is None


# ── find_judge_results ───────────────────────────────────────────────────


class TestFindJudgeResults:
    def test_picks_up_v2_aggrefact(self, tmp_path):
        (tmp_path / "factcg.json").write_text(json.dumps(_aggrefact_eval_v2()))
        out = find_judge_results(tmp_path)
        assert len(out) == 1
        label, payload = out[0]
        assert "FactCG-DeBERTa" in label
        assert payload["global_balanced_accuracy"] == pytest.approx(0.7558)

    def test_picks_up_routed_gemma(self, tmp_path):
        (tmp_path / "gemma_routed.json").write_text(json.dumps(_routed_gemma_v2()))
        out = find_judge_results(tmp_path)
        assert len(out) == 1
        label, payload = out[0]
        # Path is reduced to basename for readability.
        assert "gemma-4-E4B-Q6.gguf" in label
        assert payload["global_balanced_accuracy"] == pytest.approx(0.8211)

    def test_rejects_unrelated_json(self, tmp_path):
        (tmp_path / "noise.json").write_text('{"unrelated": true}')
        assert find_judge_results(tmp_path) == []

    def test_rejects_payload_missing_labels(self, tmp_path):
        bad = _aggrefact_eval_v2()
        del bad["labels"]
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        assert find_judge_results(tmp_path) == []

    def test_rejects_payload_missing_datasets(self, tmp_path):
        bad = _aggrefact_eval_v2()
        del bad["datasets_per_sample"]
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        assert find_judge_results(tmp_path) == []

    def test_handles_multiple_judges(self, tmp_path):
        (tmp_path / "fact.json").write_text(json.dumps(_aggrefact_eval_v2()))
        (tmp_path / "gemma.json").write_text(json.dumps(_routed_gemma_v2()))
        out = find_judge_results(tmp_path)
        assert len(out) == 2

    def test_skips_malformed_json(self, tmp_path):
        (tmp_path / "ok.json").write_text(json.dumps(_aggrefact_eval_v2()))
        (tmp_path / "bad.json").write_text("{invalid")
        out = find_judge_results(tmp_path)
        assert len(out) == 1


# ── format_judge_table ───────────────────────────────────────────────────


class TestFormatJudgeTable:
    def test_empty_returns_note(self):
        out = format_judge_table([])
        assert any("No AggreFact judge JSONs" in line for line in out)

    def test_renders_each_judge(self):
        judges = [
            ("FactCG", _aggrefact_eval_v2()),
            ("Gemma 4 routed", _routed_gemma_v2()),
        ]
        out = "\n".join(format_judge_table(judges))
        assert "FactCG" in out
        assert "Gemma 4 routed" in out
        assert "75.58%" in out
        assert "82.11%" in out
        # Per-dataset avg only present on the FactCG payload.
        assert "77.76%" in out

    def test_missing_per_dataset_avg_renders_dash(self):
        judges = [("Gemma 4 routed", _routed_gemma_v2())]
        out = "\n".join(format_judge_table(judges))
        assert " — " in out  # routed gemma has no per_dataset_avg field


# ── format_rust_table ────────────────────────────────────────────────────


class TestFormatRustTable:
    def test_empty_payload_returns_note(self):
        out = format_rust_table(None)
        assert any("rust_compute_bench.json" in line for line in out)

    def test_renders_speedups(self):
        out = "\n".join(format_rust_table(_rust_compute_bench()))
        assert "sanitizer_score" in out
        assert "27.0×" in out
        assert "21.2×" in out

    def test_includes_iteration_count(self):
        out = "\n".join(format_rust_table(_rust_compute_bench()))
        assert "5000" in out


# ── format_external_leaderboard ──────────────────────────────────────────


class TestFormatExternalLeaderboard:
    def test_empty_returns_note(self):
        out = format_external_leaderboard(None)
        assert any("external comparison" in line for line in out)

    def test_renders_entries(self):
        out = "\n".join(format_external_leaderboard(_external_leaderboard()["entries"]))
        assert "FaithLens 8B" in out
        assert "86.40%" in out
        assert "Bespoke-MiniCheck-7B" in out
        assert "77.80%" in out
        assert "https://example.org/faithlens" in out


# ── format_oracle_section ────────────────────────────────────────────────


class TestFormatOracleSection:
    def test_empty_returns_note(self):
        out = format_oracle_section(None)
        assert any("sentinel-judge analyser" in line for line in out)

    def test_labels_oracle_as_upper_bound(self):
        out = "\n".join(format_oracle_section(_sentinel_payload()))
        assert "theoretical" in out.lower()
        assert "upper bound" in out.lower()
        assert "Oracle (upper bound)" in out
        assert "86.00%" in out
        # Critical anti-fabrication: never imply Oracle is a deployable
        # leaderboard system.
        assert "matches global SOTA" not in out
        assert "matches the global SOTA" not in out


# ── generate_report end-to-end ───────────────────────────────────────────


class TestGenerateReport:
    def _setup_results_dir(self, tmp_path: Path) -> Path:
        results = tmp_path / "results"
        results.mkdir()
        (results / "factcg.json").write_text(json.dumps(_aggrefact_eval_v2()))
        (results / "gemma.json").write_text(json.dumps(_routed_gemma_v2()))
        (results / "rust_compute_bench.json").write_text(
            json.dumps(_rust_compute_bench())
        )
        return results

    def _setup_leaderboard(self, tmp_path: Path) -> Path:
        path = tmp_path / "leaderboard.json"
        path.write_text(json.dumps(_external_leaderboard()))
        return path

    def _setup_sentinel(self, tmp_path: Path) -> Path:
        path = tmp_path / "sentinel.json"
        path.write_text(json.dumps(_sentinel_payload()))
        return path

    def test_writes_file(self, tmp_path):
        results = self._setup_results_dir(tmp_path)
        out = tmp_path / "report.md"
        ret = generate_report(results, out)
        assert ret == out
        assert out.exists()
        text = out.read_text()
        assert text.startswith("# Director-AI — AggreFact & Compute Benchmark Summary")

    def test_contains_every_judge(self, tmp_path):
        results = self._setup_results_dir(tmp_path)
        out = tmp_path / "report.md"
        generate_report(results, out)
        text = out.read_text()
        assert "FactCG-DeBERTa" in text
        assert "gemma-4-E4B-Q6.gguf" in text
        assert "75.58%" in text
        assert "82.11%" in text

    def test_contains_rust_section(self, tmp_path):
        results = self._setup_results_dir(tmp_path)
        out = tmp_path / "report.md"
        generate_report(results, out)
        text = out.read_text()
        assert "sanitizer_score" in text
        assert "27.0×" in text

    def test_external_leaderboard_optional(self, tmp_path):
        results = self._setup_results_dir(tmp_path)
        out = tmp_path / "report.md"
        generate_report(results, out)  # no leaderboard arg
        text = out.read_text()
        assert "external comparison" in text  # the omission note

    def test_external_leaderboard_renders_when_provided(self, tmp_path):
        results = self._setup_results_dir(tmp_path)
        leaderboard = self._setup_leaderboard(tmp_path)
        out = tmp_path / "report.md"
        generate_report(results, out, leaderboard_path=leaderboard)
        text = out.read_text()
        assert "FaithLens 8B" in text
        assert "86.40%" in text

    def test_sentinel_optional(self, tmp_path):
        results = self._setup_results_dir(tmp_path)
        out = tmp_path / "report.md"
        generate_report(results, out)
        text = out.read_text()
        assert "sentinel-judge analyser" in text  # the omission note

    def test_sentinel_renders_when_provided(self, tmp_path):
        results = self._setup_results_dir(tmp_path)
        sentinel = self._setup_sentinel(tmp_path)
        out = tmp_path / "report.md"
        generate_report(results, out, sentinel_path=sentinel)
        text = out.read_text()
        assert "Oracle (upper bound)" in text
        assert "86.00%" in text

    def test_no_banned_words_in_output(self, tmp_path):
        """Anti-slop / Tier-0 fabrication regression guard.

        Anything in this list would have shown up in the original
        20d9ccb master report. The new generator must NEVER emit them.
        """
        results = self._setup_results_dir(tmp_path)
        leaderboard = self._setup_leaderboard(tmp_path)
        sentinel = self._setup_sentinel(tmp_path)
        out = tmp_path / "report.md"
        generate_report(
            results,
            out,
            leaderboard_path=leaderboard,
            sentinel_path=sentinel,
        )
        text = out.read_text()
        banned = [
            "comprehensive",
            "Comprehensive",
            "v4.0.0-rc1",
            "Release Candidate",
            "matches the world SOTA",
            "matches the current global SOTA",
            "Strategic Insight",
            "$0",
            "$$$",
            "$100/mo",
            "Pay-per-token",
            "~65%",
            "~76%",
        ]
        for word in banned:
            assert word not in text, f"banned token leaked: {word!r}"

    def test_handles_empty_results_dir(self, tmp_path):
        results = tmp_path / "empty"
        results.mkdir()
        out = tmp_path / "report.md"
        generate_report(results, out)
        text = out.read_text()
        assert "No AggreFact judge JSONs" in text
        assert "rust_compute_bench.json" in text  # missing-rust note
        # The header is still present.
        assert text.startswith("# Director-AI — AggreFact & Compute Benchmark Summary")

    def test_no_hardcoded_speedup_when_rust_missing(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        (results / "factcg.json").write_text(json.dumps(_aggrefact_eval_v2()))
        out = tmp_path / "report.md"
        generate_report(results, out)
        text = out.read_text()
        assert "9.4×" not in text  # used to be hardcoded — must not be
        assert "27×" not in text
        assert "21×" not in text
        assert "33×" not in text
