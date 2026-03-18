# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Fine-tuning Pipeline Tests

from __future__ import annotations

import json

import pytest


class TestFinetuneConfig:
    def test_defaults(self):
        from director_ai.core.finetune import FinetuneConfig

        cfg = FinetuneConfig()
        assert cfg.epochs == 3
        assert cfg.batch_size == 16
        assert cfg.learning_rate == 2e-5
        assert cfg.max_length == 512
        assert "FactCG" in cfg.base_model

    def test_custom_values(self):
        from director_ai.core.finetune import FinetuneConfig

        cfg = FinetuneConfig(epochs=5, learning_rate=1e-5, output_dir="/tmp/test")
        assert cfg.epochs == 5
        assert cfg.learning_rate == 1e-5
        assert cfg.output_dir == "/tmp/test"

    def test_phase_e_defaults(self):
        from director_ai.core.finetune import FinetuneConfig

        cfg = FinetuneConfig()
        assert cfg.mix_general_data is False
        assert cfg.general_data_path is None
        assert cfg.general_data_ratio == 0.2
        assert cfg.early_stopping_patience == 0
        assert cfg.class_weighted_loss is False
        assert cfg.auto_benchmark is False
        assert cfg.auto_onnx_export is False

    def test_phase_e_custom(self):
        from director_ai.core.finetune import FinetuneConfig

        cfg = FinetuneConfig(
            mix_general_data=True,
            general_data_ratio=0.3,
            early_stopping_patience=5,
            class_weighted_loss=True,
            auto_benchmark=True,
        )
        assert cfg.mix_general_data is True
        assert cfg.general_data_ratio == 0.3
        assert cfg.early_stopping_patience == 5
        assert cfg.class_weighted_loss is True
        assert cfg.auto_benchmark is True


class TestFinetuneResult:
    def test_defaults(self):
        from director_ai.core.finetune import FinetuneResult

        result = FinetuneResult()
        assert result.epochs_completed == 0
        assert result.best_balanced_accuracy == 0.0
        assert result.eval_metrics == {}
        assert result.regression_report == {}
        assert result.onnx_path == ""
        assert result.mixed_general_samples == 0

    def test_with_values(self):
        from director_ai.core.finetune import FinetuneResult

        result = FinetuneResult(
            output_dir="./model",
            epochs_completed=3,
            train_samples=1000,
            best_balanced_accuracy=0.82,
        )
        assert result.output_dir == "./model"
        assert result.best_balanced_accuracy == 0.82


class TestLoadJsonl:
    def test_load_standard_format(self, tmp_path):
        from director_ai.core.finetune import _load_jsonl

        f = tmp_path / "train.jsonl"
        f.write_text(
            json.dumps(
                {
                    "premise": "The sky is blue.",
                    "hypothesis": "Sky is blue.",
                    "label": 1,
                },
            )
            + "\n"
            + json.dumps(
                {"premise": "Cats are dogs.", "hypothesis": "Cats bark.", "label": 0},
            )
            + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert len(rows) == 2
        assert rows[0]["label"] == 1
        assert rows[1]["label"] == 0

    def test_load_alternative_field_names(self, tmp_path):
        from director_ai.core.finetune import _load_jsonl

        f = tmp_path / "train.jsonl"
        f.write_text(
            json.dumps({"doc": "Source text.", "claim": "Derived claim.", "label": 1})
            + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert len(rows) == 1
        assert rows[0]["premise"] == "Source text."
        assert rows[0]["hypothesis"] == "Derived claim."

    def test_skip_incomplete_rows(self, tmp_path):
        from director_ai.core.finetune import _load_jsonl

        f = tmp_path / "train.jsonl"
        f.write_text(
            json.dumps({"premise": "ok", "hypothesis": "ok", "label": 1})
            + "\n"
            + json.dumps({"premise": "missing hypothesis"})
            + "\n"
            + json.dumps({"hypothesis": "missing premise", "label": 0})
            + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert len(rows) == 1

    def test_skip_blank_lines(self, tmp_path):
        from director_ai.core.finetune import _load_jsonl

        f = tmp_path / "train.jsonl"
        f.write_text(
            "\n"
            + json.dumps({"premise": "a", "hypothesis": "b", "label": 1})
            + "\n"
            + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert len(rows) == 1

    def test_empty_file_returns_empty(self, tmp_path):
        from director_ai.core.finetune import _load_jsonl

        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        rows = _load_jsonl(f)
        assert rows == []


class TestMixGeneralData:
    def _make_jsonl(self, tmp_path, name, n):
        rows = [
            {"premise": f"P{i}", "hypothesis": f"H{i}", "label": i % 2}
            for i in range(n)
        ]
        f = tmp_path / name
        f.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )
        return f

    def test_mix_adds_general_data(self, tmp_path):
        from director_ai.core.finetune import _mix_general_data

        domain = [
            {"premise": f"D{i}", "hypothesis": f"C{i}", "label": i % 2}
            for i in range(100)
        ]
        general_file = self._make_jsonl(tmp_path, "general.jsonl", 50)
        mixed, n_added = _mix_general_data(
            domain,
            str(general_file),
            ratio=0.2,
            seed=42,
        )
        assert n_added > 0
        assert len(mixed) == len(domain) + n_added

    def test_mix_with_missing_file_returns_original(self):
        from director_ai.core.finetune import _mix_general_data

        domain = [{"premise": "D", "hypothesis": "C", "label": 1}]
        mixed, n_added = _mix_general_data(
            domain,
            "/nonexistent_path_xyz.jsonl",
            0.2,
            42,
        )
        assert n_added == 0
        assert mixed is domain

    def test_mix_ratio_is_approximate(self, tmp_path):
        from director_ai.core.finetune import _mix_general_data

        domain = [
            {"premise": f"D{i}", "hypothesis": f"C{i}", "label": i % 2}
            for i in range(800)
        ]
        general_file = self._make_jsonl(tmp_path, "general.jsonl", 500)
        mixed, n_added = _mix_general_data(
            domain,
            str(general_file),
            ratio=0.2,
            seed=42,
        )
        actual_ratio = n_added / len(mixed)
        assert 0.15 < actual_ratio < 0.25


class TestComputeClassWeights:
    def test_balanced_weights_are_equal(self):
        from director_ai.core.finetune import _compute_class_weights

        rows = [{"label": 0}] * 100 + [{"label": 1}] * 100
        weights = _compute_class_weights(rows)
        assert len(weights) == 2
        assert abs(weights[0] - weights[1]) < 1e-6

    def test_imbalanced_weights_compensate(self):
        from director_ai.core.finetune import _compute_class_weights

        rows = [{"label": 0}] * 900 + [{"label": 1}] * 100
        weights = _compute_class_weights(rows)
        assert weights[1] > weights[0]
        assert weights[1] / weights[0] > 5


class TestExports:
    def test_finetune_in_core_all(self):
        from director_ai.core import __all__

        assert "finetune_nli" in __all__
        assert "FinetuneConfig" in __all__
        assert "FinetuneResult" in __all__

    def test_importable(self):
        from director_ai.core import FinetuneConfig, FinetuneResult, finetune_nli

        assert callable(finetune_nli)
        assert FinetuneConfig is not None
        assert FinetuneResult is not None


class TestCliFinetune:
    def test_finetune_no_args_exits(self):
        from director_ai.cli import main

        with pytest.raises(SystemExit):
            main(["finetune"])

    def test_finetune_missing_file_exits(self):
        from director_ai.cli import main

        with pytest.raises(SystemExit):
            main(["finetune", "/nonexistent_finetune_test_xyz/train.jsonl"])
