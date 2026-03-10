# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Fine-tuning Pipeline Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

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


class TestFinetuneResult:
    def test_defaults(self):
        from director_ai.core.finetune import FinetuneResult

        result = FinetuneResult()
        assert result.epochs_completed == 0
        assert result.best_balanced_accuracy == 0.0
        assert result.eval_metrics == {}

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
            json.dumps({"premise": "The sky is blue.", "hypothesis": "Sky is blue.", "label": 1}) + "\n"
            + json.dumps({"premise": "Cats are dogs.", "hypothesis": "Cats bark.", "label": 0}) + "\n",
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
            json.dumps({"doc": "Source text.", "claim": "Derived claim.", "label": 1}) + "\n",
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
            json.dumps({"premise": "ok", "hypothesis": "ok", "label": 1}) + "\n"
            + json.dumps({"premise": "missing hypothesis"}) + "\n"
            + json.dumps({"hypothesis": "missing premise", "label": 0}) + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert len(rows) == 1

    def test_skip_blank_lines(self, tmp_path):
        from director_ai.core.finetune import _load_jsonl

        f = tmp_path / "train.jsonl"
        f.write_text(
            "\n"
            + json.dumps({"premise": "a", "hypothesis": "b", "label": 1}) + "\n"
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
