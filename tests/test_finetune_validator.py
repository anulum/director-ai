# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Fine-tuning Data Validator Tests

from __future__ import annotations

import json

from director_ai.core.finetune_validator import (
    DataQualityReport,
    validate_finetune_data,
)


def _make_samples(n_pos: int, n_neg: int) -> list[dict]:
    rows = []
    for i in range(n_pos):
        rows.append({"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": 1})
    for i in range(n_neg):
        rows.append(
            {"premise": f"Source {i}.", "hypothesis": f"Wrong {i}.", "label": 0},
        )
    return rows


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


class TestDataQualityReport:
    def test_defaults_valid(self):
        r = DataQualityReport()
        assert r.is_valid
        assert r.total_samples == 0

    def test_errors_make_invalid(self):
        r = DataQualityReport(errors=["bad"])
        assert not r.is_valid

    def test_summary_contains_key_fields(self):
        r = DataQualityReport(total_samples=100, label_distribution={0: 50, 1: 50})
        s = r.summary()
        assert "100" in s
        assert "Valid: True" in s


class TestValidatePass:
    def test_valid_dataset(self, tmp_path):
        rows = _make_samples(300, 300)
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert report.is_valid
        assert report.total_samples == 600
        assert report.label_distribution == {1: 300, 0: 300}
        assert report.class_balance_ratio == 1.0
        assert report.duplicate_count == 0

    def test_cost_estimate_populated(self, tmp_path):
        rows = _make_samples(300, 300)
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f, epochs=3)
        assert report.estimated_train_time_min > 0
        assert report.estimated_cost_usd > 0


class TestValidateFail:
    def test_file_not_found(self, tmp_path):
        report = validate_finetune_data(tmp_path / "nonexistent.jsonl")
        assert not report.is_valid
        assert "not found" in report.errors[0].lower()

    def test_too_few_samples(self, tmp_path):
        rows = _make_samples(50, 50)
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert not report.is_valid
        assert any("500" in e for e in report.errors)

    def test_too_few_per_class(self, tmp_path):
        rows = _make_samples(400, 50)
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert not report.is_valid
        assert any("100" in e for e in report.errors)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        report = validate_finetune_data(f)
        assert not report.is_valid

    def test_no_valid_samples(self, tmp_path):
        rows = [{"premise": "a"}, {"hypothesis": "b"}]
        f = tmp_path / "bad.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert not report.is_valid
        assert any("no valid" in e.lower() for e in report.errors)

    def test_invalid_labels(self, tmp_path):
        rows = [{"premise": "a", "hypothesis": "b", "label": 5} for _ in range(100)]
        f = tmp_path / "bad_labels.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert not report.is_valid


class TestValidateWarnings:
    def test_class_imbalance_warning(self, tmp_path):
        # Need ratio below 1/5.0 = 0.2; 600:101 = 0.168
        rows = _make_samples(600, 101)
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert any("imbalance" in w.lower() for w in report.warnings)

    def test_duplicate_warning(self, tmp_path):
        base = _make_samples(250, 250)
        dupes = base[:100]
        rows = base + dupes
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert report.duplicate_count == 100

    def test_parse_error_warning(self, tmp_path):
        rows = _make_samples(300, 300)
        f = tmp_path / "train.jsonl"
        lines = [json.dumps(r) for r in rows]
        lines.append("not valid json {{{{")
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = validate_finetune_data(f)
        assert report.parse_error_count == 1
        assert any("parse" in w.lower() for w in report.warnings)

    def test_empty_field_warning(self, tmp_path):
        rows = _make_samples(300, 300)
        rows.append({"premise": "", "hypothesis": "something", "label": 1})
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert report.empty_field_count == 1

    def test_low_per_class_warning(self, tmp_path):
        rows = _make_samples(150, 350)
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert any("recommended" in w.lower() for w in report.warnings)


class TestValidateFieldAliases:
    def test_doc_claim_aliases(self, tmp_path):
        rows = [
            {"doc": f"Doc {i}.", "claim": f"Claim {i}.", "label": i % 2}
            for i in range(600)
        ]
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert report.is_valid
        assert report.total_samples == 600

    def test_context_response_aliases(self, tmp_path):
        rows = [
            {"context": f"Ctx {i}.", "response": f"Resp {i}.", "label": i % 2}
            for i in range(600)
        ]
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert report.is_valid


class TestSummaryOutput:
    def test_summary_with_warnings(self):
        r = DataQualityReport(total_samples=500, warnings=["imbalance"])
        s = r.summary()
        assert "Warnings: 1" in s

    def test_summary_with_errors(self):
        r = DataQualityReport(total_samples=500, errors=["too few"])
        s = r.summary()
        assert "Errors: 1" in s
        assert "Valid: False" in s


class TestLabelEdgeCases:
    def test_label_none_counted_as_empty(self, tmp_path):
        rows = _make_samples(300, 300)
        rows.append({"premise": "text", "hypothesis": "claim"})
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert report.empty_field_count == 1

    def test_label_non_numeric_counted_as_parse_error(self, tmp_path):
        rows = _make_samples(300, 300)
        rows.append({"premise": "text", "hypothesis": "claim", "label": "maybe"})
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert report.parse_error_count == 1

    def test_too_many_bad_labels_truncates_errors(self, tmp_path):
        rows = [
            {"premise": f"P{i}", "hypothesis": f"H{i}", "label": 99} for i in range(20)
        ]
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert any("truncated" in e for e in report.errors)

    def test_long_text_warning(self, tmp_path):
        rows = _make_samples(300, 300)
        rows.append({"premise": "word " * 5000, "hypothesis": "claim", "label": 1})
        f = tmp_path / "train.jsonl"
        _write_jsonl(f, rows)
        report = validate_finetune_data(f)
        assert any("truncated" in w.lower() for w in report.warnings)


class TestExports:
    def test_importable_from_core(self):
        from director_ai.core import DataQualityReport, validate_finetune_data

        assert callable(validate_finetune_data)
        assert DataQualityReport is not None
