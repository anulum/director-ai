# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tuning Pipeline Tests
"""Multi-angle tests for fine-tuning pipeline."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from director_ai.core.training.finetune import TrainingRow
from director_ai.core.training.training_provenance import (
    PROVENANCE_FILENAME,
    TrainingProvenance,
)


class TestFinetuneConfig:
    """FinetuneConfig defaults and overrides."""

    def test_defaults(self) -> None:
        """Core hyperparameters have their documented defaults."""
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig()
        assert cfg.epochs == 3
        assert cfg.batch_size == 16
        assert cfg.learning_rate == 2e-5
        assert cfg.max_length == 512
        assert "FactCG" in cfg.base_model

    def test_custom_values(self) -> None:
        """Constructor arguments override the defaults."""
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig(epochs=5, learning_rate=1e-5, output_dir="/tmp/test")
        assert cfg.epochs == 5
        assert cfg.learning_rate == 1e-5
        assert cfg.output_dir == "/tmp/test"

    def test_phase_e_defaults(self) -> None:
        """Replay, early stopping and benchmark options start disabled."""
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig()
        assert cfg.mix_general_data is False
        assert cfg.general_data_path is None
        assert cfg.general_data_ratio == 0.2
        assert cfg.early_stopping_patience == 0
        assert cfg.class_weighted_loss is False
        assert cfg.auto_benchmark is False
        assert cfg.auto_onnx_export is False

    def test_phase_e_custom(self) -> None:
        """Replay, early stopping and benchmark options can be enabled."""
        from director_ai.core.training.finetune import FinetuneConfig

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
    """FinetuneResult defaults and values."""

    def test_defaults(self) -> None:
        """An empty result carries zero metrics and no reports."""
        from director_ai.core.training.finetune import FinetuneResult

        result = FinetuneResult()
        assert result.epochs_completed == 0
        assert result.best_balanced_accuracy == 0.0
        assert result.eval_metrics == {}
        assert result.regression_report == {}
        assert result.onnx_path == ""
        assert result.mixed_general_samples == 0

    def test_with_values(self) -> None:
        """Result fields keep the supplied values."""
        from director_ai.core.training.finetune import FinetuneResult

        result = FinetuneResult(
            output_dir="./model",
            epochs_completed=3,
            train_samples=1000,
            best_balanced_accuracy=0.82,
        )
        assert result.output_dir == "./model"
        assert result.best_balanced_accuracy == 0.82


class TestLoadJsonl:
    """JSONL loading of NLI training rows."""

    def test_load_standard_format(self, tmp_path: Path) -> None:
        """Premise, hypothesis and label rows load in order."""
        from director_ai.core.training.finetune import _load_jsonl

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

    def test_load_alternative_field_names(self, tmp_path: Path) -> None:
        """The doc and claim aliases map to premise and hypothesis."""
        from director_ai.core.training.finetune import _load_jsonl

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

    def test_skip_incomplete_rows(self, tmp_path: Path) -> None:
        """Rows missing a required field are skipped."""
        from director_ai.core.training.finetune import _load_jsonl

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

    def test_skip_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines are skipped."""
        from director_ai.core.training.finetune import _load_jsonl

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

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        """An empty file loads no rows."""
        from director_ai.core.training.finetune import _load_jsonl

        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        rows = _load_jsonl(f)
        assert rows == []

    def test_load_keeps_a_valid_source_cluster(self, tmp_path: Path) -> None:
        """A valid source cluster stays on the loaded row."""
        from director_ai.core.training.finetune import _load_jsonl

        f = tmp_path / "train.jsonl"
        f.write_text(
            json.dumps(
                {
                    "premise": "a",
                    "hypothesis": "b",
                    "label": 1,
                    "source_cluster": "doc-3",
                }
            )
            + "\n"
            + json.dumps({"premise": "c", "hypothesis": "d", "label": 0})
            + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert rows[0]["source_cluster"] == "doc-3"
        assert "source_cluster" not in rows[1]

    def test_load_refuses_a_malformed_source_cluster(self, tmp_path: Path) -> None:
        """A padded source cluster is refused."""
        from director_ai.core.training.finetune import _load_jsonl

        f = tmp_path / "train.jsonl"
        f.write_text(
            json.dumps(
                {"premise": "a", "hypothesis": "b", "label": 1, "source_cluster": " x"}
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="source_cluster"):
            _load_jsonl(f)


class TestMixGeneralData:
    """Explicit replay mixing and its refusals."""

    def _make_jsonl(self, tmp_path: Path, name: str, n: int) -> Path:
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

    def test_mix_adds_general_data(self, tmp_path: Path) -> None:
        """Replay rows are added to the domain rows."""
        from director_ai.core.training.finetune import _mix_general_data

        domain: list[TrainingRow] = [
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

    def test_mix_refuses_a_missing_replay_file(self) -> None:
        """A missing replay file raises instead of skipping replay."""
        from director_ai.core.training.finetune import _mix_general_data

        domain: list[TrainingRow] = [{"premise": "D", "hypothesis": "C", "label": 1}]
        with pytest.raises(FileNotFoundError, match="replay data not found"):
            _mix_general_data(domain, "/nonexistent_path_xyz.jsonl", 0.2, 42)

    def test_mix_ratio_is_approximate(self, tmp_path: Path) -> None:
        """The replay share approximates the requested ratio."""
        from director_ai.core.training.finetune import _mix_general_data

        domain: list[TrainingRow] = [
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

    def test_finetune_refuses_replay_mixing_without_a_path(
        self, tmp_path: Path
    ) -> None:
        """Replay mixing without a path is refused before training."""
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        train_file = self._make_jsonl(tmp_path, "train.jsonl", 4)
        with pytest.raises(ValueError, match="general_data_path"):
            finetune_nli(train_file, config=FinetuneConfig(mix_general_data=True))

    def test_mix_uses_all_general_rows_when_ratio_exceeds_available(
        self, tmp_path: Path
    ) -> None:
        """A ratio above the available rows uses every replay row."""
        from director_ai.core.training.finetune import _mix_general_data

        domain: list[TrainingRow] = [
            {"premise": f"D{i}", "hypothesis": f"C{i}", "label": i % 2}
            for i in range(20)
        ]
        general_file = self._make_jsonl(tmp_path, "general.jsonl", 3)

        mixed, n_added = _mix_general_data(domain, general_file, ratio=0.5, seed=42)

        assert n_added == 3
        assert len(mixed) == 23

    def test_mix_refuses_an_empty_replay_file(self, tmp_path: Path) -> None:
        """An empty replay file raises instead of skipping replay."""
        from director_ai.core.training.finetune import _mix_general_data

        domain: list[TrainingRow] = [{"premise": "D", "hypothesis": "C", "label": 1}]
        general_file = tmp_path / "empty.jsonl"
        general_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="No valid samples"):
            _mix_general_data(domain, general_file, 0.2, 42)


class TestComputeClassWeights:
    """Inverse-frequency class weights."""

    def test_balanced_weights_are_equal(self) -> None:
        """Balanced labels get equal weights."""
        from director_ai.core.training.finetune import _compute_class_weights

        zero: TrainingRow = {"label": 0}
        one: TrainingRow = {"label": 1}
        rows = [zero] * 100 + [one] * 100
        weights = _compute_class_weights(rows)
        assert len(weights) == 2
        assert abs(weights[0] - weights[1]) < 1e-6

    def test_imbalanced_weights_compensate(self) -> None:
        """The minority class gets the larger weight."""
        from director_ai.core.training.finetune import _compute_class_weights

        zero: TrainingRow = {"label": 0}
        one: TrainingRow = {"label": 1}
        rows = [zero] * 900 + [one] * 100
        weights = _compute_class_weights(rows)
        assert weights[1] > weights[0]
        assert weights[1] / weights[0] > 5


class TestDatasetAndMetrics:
    """Dataset preparation, metrics and weighted loss."""

    def test_prepare_dataset_factcg_and_pair_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FactCG and paired inputs tokenise through the expected columns."""
        from director_ai.core.training.finetune import _prepare_dataset

        class _Dataset:
            def __init__(self, data: dict[str, Any]) -> None:
                self.data = data
                self.map_calls: list[dict[str, Any]] = []
                self.format_name: str | None = None

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> _Dataset:
                return cls(data)

            def map(
                self,
                fn: Any,
                *,
                batched: bool,
                batch_size: int,
                remove_columns: list[str],
            ) -> _Dataset:
                self.map_calls.append(
                    {
                        "batched": batched,
                        "batch_size": batch_size,
                        "remove_columns": remove_columns,
                    }
                )
                self.data.update(fn(self.data))
                return self

            def set_format(self, format_name: str) -> None:
                self.format_name = format_name

        datasets_module = types.SimpleNamespace(Dataset=_Dataset)
        monkeypatch.setitem(sys.modules, "datasets", datasets_module)

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def tokenizer(*args: Any, **kwargs: Any) -> dict[str, list[list[int]]]:
            calls.append((args, kwargs))
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

        rows: list[TrainingRow] = [
            {"premise": "Premise", "hypothesis": "Claim", "label": 1}
        ]
        factcg = _prepare_dataset(rows, tokenizer, max_length=17, is_factcg=True)
        paired = _prepare_dataset(rows, tokenizer, max_length=19, is_factcg=False)

        assert "Choose your answer" in factcg.data["text"][0]
        assert factcg.data["labels"] == [1]
        assert factcg.map_calls[0]["remove_columns"] == ["text"]
        assert factcg.format_name == "torch"
        assert paired.data["premise"] == ["Premise"]
        assert paired.data["hypothesis"] == ["Claim"]
        assert paired.map_calls[0]["remove_columns"] == ["premise", "hypothesis"]
        assert paired.format_name == "torch"
        assert calls[0][1]["max_length"] == 17
        assert calls[1][1]["max_length"] == 19

    def test_metrics_cover_empty_balanced_accuracy_and_binary_f1_edges(self) -> None:
        """Metric helpers handle empty and edge inputs."""
        import numpy as np

        from director_ai.core.training.finetune import (
            _balanced_accuracy,
            _binary_f1_score,
            _compute_metrics,
        )

        assert _balanced_accuracy([], []) == 0.0
        assert _balanced_accuracy([0, 1, 1], [0, 0, 1]) == pytest.approx(0.75)
        assert _binary_f1_score([0, 0], [1, 0]) == 0.0
        assert _binary_f1_score([1, 1, 0], [1, 0, 1]) == pytest.approx(0.5)

        metrics = _compute_metrics(
            (
                np.asarray([[0.1, 0.9], [0.8, 0.2], [0.2, 0.7]]),
                np.asarray([1, 0, 1]),
            )
        )
        assert metrics == {"balanced_accuracy": 1.0, "f1": 1.0}

    def test_metric_defensive_zero_denominator_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero denominators yield zero metrics."""
        from director_ai.core.training.finetune import (
            _balanced_accuracy,
            _binary_f1_score,
        )

        class _Labels:
            def __eq__(self, other: object) -> Any:
                assert other == "ghost"
                return "empty-mask"

        class _NpForBalancedAccuracy:
            @staticmethod
            def asarray(values: Any) -> _Labels:
                return _Labels()

            @staticmethod
            def unique(values: Any) -> list[str]:
                return ["ghost"]

            @staticmethod
            def count_nonzero(values: Any) -> int:
                assert values == "empty-mask"
                return 0

        monkeypatch.setitem(sys.modules, "numpy", _NpForBalancedAccuracy)
        assert _balanced_accuracy(["label"], ["pred"]) == 0.0

        class _NpForF1:
            calls = iter((1, -1, -1))

            @staticmethod
            def asarray(values: Any) -> int:
                return 1

            @classmethod
            def count_nonzero(cls, values: Any) -> int:
                return next(cls.calls)

        monkeypatch.setitem(sys.modules, "numpy", _NpForF1)
        assert _binary_f1_score([1], [1]) == 0.0

    def test_python_sum_and_mean_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Python reducers match the expected sums and means."""
        from director_ai.core.training import finetune as finetune_mod

        monkeypatch.setattr(finetune_mod, "_RUST_FINETUNE", False)

        assert finetune_mod._sum_int([1, 2, 3]) == 6
        assert finetune_mod._sum_float([0.1, 0.2]) == pytest.approx(0.3)
        assert finetune_mod._mean_float([]) == 0.0
        assert finetune_mod._mean_float([0.25, 0.75]) == pytest.approx(0.5)

    def test_weighted_trainer_compute_loss_returns_loss_or_outputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The weighted trainer returns the loss, with outputs on request."""
        from director_ai.core.training.finetune import _make_weighted_trainer_class

        class _WeightTensor:
            def to(self, device: str) -> _WeightTensor:
                assert device == "cpu"
                return self

        class _CrossEntropyLoss:
            def __init__(self, weight: Any) -> None:
                self.weight = weight

            def __call__(self, logits: Any, labels: Any) -> str:
                assert logits == SimpleNamespace(device="cpu")
                assert labels == ["labels"]
                return "weighted-loss"

        class _NN:
            CrossEntropyLoss = _CrossEntropyLoss

        class _Torch:
            float32 = "float32"
            nn = _NN

            @staticmethod
            def tensor(values: list[float], dtype: str) -> _WeightTensor:
                assert values == [0.5, 2.0]
                assert dtype == "float32"
                return _WeightTensor()

        class _Trainer:
            pass

        monkeypatch.setitem(sys.modules, "torch", _Torch)
        monkeypatch.setitem(
            sys.modules, "transformers", SimpleNamespace(Trainer=_Trainer)
        )
        trainer_cls = _make_weighted_trainer_class([0.5, 2.0])

        class _Model:
            def __call__(self, **inputs: Any) -> SimpleNamespace:
                assert inputs == {"input_ids": [1]}
                return SimpleNamespace(logits=SimpleNamespace(device="cpu"))

        trainer = trainer_cls()
        inputs = {"labels": ["labels"], "input_ids": [1]}
        assert trainer.compute_loss(_Model(), dict(inputs)) == "weighted-loss"
        assert trainer.compute_loss(_Model(), dict(inputs), return_outputs=True) == (
            "weighted-loss",
            SimpleNamespace(logits=SimpleNamespace(device="cpu")),
        )


class TestFinetuneNli:
    """finetune_nli orchestration with a fake training stack."""

    def _write_rows(self, path: Path, rows: list[dict[str, Any]]) -> Path:
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def _install_fake_transformers(
        self, monkeypatch: pytest.MonkeyPatch, recorder: dict[str, Any]
    ) -> None:
        class _Tokenizer:
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> _Tokenizer:
                recorder["tokenizer_from_pretrained"] = (args, kwargs)
                return cls()

            def save_pretrained(self, output_dir: str) -> None:
                recorder["tokenizer_saved"] = output_dir

        class _Model:
            def __init__(self) -> None:
                self.to_calls: list[str] = []

            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> _Model:
                recorder["model_from_pretrained"] = (args, kwargs)
                model = cls()
                recorder["model"] = model
                return model

            def to(self, device: str) -> _Model:
                self.to_calls.append(device)
                return self

        class _TrainingArguments:
            def __init__(self, **kwargs: Any) -> None:
                recorder["training_args"] = kwargs

        class _EarlyStoppingCallback:
            def __init__(self, **kwargs: Any) -> None:
                recorder["early_stopping"] = kwargs

        class _Trainer:
            def __init__(self, **kwargs: Any) -> None:
                recorder["trainer"] = self
                recorder["trainer_kwargs"] = kwargs
                self.model = kwargs["model"]

            def train(self) -> SimpleNamespace:
                recorder["trained"] = True
                if recorder.get("drop_model_on_train"):
                    self.model = None
                return SimpleNamespace(training_loss=0.123)

            def save_model(self, output_dir: str) -> None:
                recorder["model_saved"] = output_dir

            def evaluate(self) -> dict[str, float]:
                return {"eval_balanced_accuracy": 0.91, "eval_f1": 0.88}

        module = SimpleNamespace(
            AutoModelForSequenceClassification=_Model,
            AutoTokenizer=_Tokenizer,
            EarlyStoppingCallback=_EarlyStoppingCallback,
            Trainer=_Trainer,
            TrainingArguments=_TrainingArguments,
        )
        monkeypatch.setitem(sys.modules, "transformers", module)

    def test_finetune_requires_valid_training_rows(self, tmp_path: Path) -> None:
        """An empty training file is refused."""
        from director_ai.core.training.finetune import finetune_nli

        train_file = tmp_path / "empty.jsonl"
        train_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="No valid samples"):
            finetune_nli(train_file)

    def test_finetune_reports_missing_extra_after_loading_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing transformers install names the extra to install."""
        from director_ai.core.training import finetune as finetune_mod
        from director_ai.core.training.finetune import finetune_nli

        train_file = self._write_rows(
            tmp_path / "train.jsonl",
            [{"premise": "P", "hypothesis": "H", "label": 1}],
        )
        monkeypatch.setattr(
            finetune_mod,
            "resolve_model_revision",
            lambda model, revision: revision,
        )
        monkeypatch.setitem(sys.modules, "transformers", None)

        with pytest.raises(ImportError, match=r"director-ai\[finetune\]"):
            finetune_nli(train_file)

    def test_finetune_full_orchestration_with_eval_callbacks_and_exports(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A full run trains, evaluates, benchmarks, exports and records provenance."""
        import director_ai.core._device as device_mod
        import director_ai.core.scoring.nli as nli_mod
        import director_ai.core.training.finetune_benchmark as benchmark_mod
        from director_ai.core.training import finetune as finetune_mod
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        recorder: dict[str, Any] = {}
        self._install_fake_transformers(monkeypatch, recorder)
        train_file = self._write_rows(
            tmp_path / "train.jsonl",
            [
                {"premise": "P1", "hypothesis": "H1", "label": 1},
                {"premise": "P2", "hypothesis": "H2", "label": 0},
            ],
        )
        eval_file = self._write_rows(
            tmp_path / "eval.jsonl",
            [{"premise": "EP", "hypothesis": "EH", "label": 1}],
        )
        general_file = self._write_rows(
            tmp_path / "general.jsonl",
            [{"premise": "GP", "hypothesis": "GH", "label": 0}],
        )

        monkeypatch.setattr(
            finetune_mod,
            "resolve_model_revision",
            lambda model, revision: "resolved-revision",
        )
        monkeypatch.setattr(
            nli_mod,
            "clear_model_cache",
            lambda: recorder.setdefault("cache_cleared", True),
        )
        monkeypatch.setattr(
            nli_mod,
            "export_onnx",
            lambda src, dst: recorder.setdefault("onnx", (src, dst)),
        )
        monkeypatch.setattr(
            device_mod,
            "release_torch_cuda",
            lambda: recorder.setdefault("released_cuda", True),
        )
        monkeypatch.setattr(
            finetune_mod,
            "_prepare_dataset",
            lambda rows, tokenizer, max_length, is_factcg: {
                "rows": list(rows),
                "max_length": max_length,
                "is_factcg": is_factcg,
            },
        )
        monkeypatch.setattr(
            finetune_mod,
            "_make_weighted_trainer_class",
            lambda weights: sys.modules["transformers"].Trainer,
        )
        benchmark_calls: list[tuple[str, str | None, Path | None]] = []

        def fake_benchmark(
            output_dir: str, general_path: str | None, eval_path: Path | None
        ) -> SimpleNamespace:
            benchmark_calls.append((output_dir, general_path, eval_path))
            return SimpleNamespace(
                recommendation="deploy",
                regression_acceptable=True,
                general_accuracy=0.81,
                domain_accuracy=0.93,
                regression_pp=-0.2,
                details={},
            )

        monkeypatch.setattr(benchmark_mod, "benchmark_finetuned_model", fake_benchmark)

        cfg = FinetuneConfig(
            base_model="local/nli",
            base_model_revision="pin",
            output_dir=str(tmp_path / "model"),
            epochs=2,
            batch_size=4,
            max_length=33,
            fp16=False,
            mix_general_data=True,
            general_data_path=str(general_file),
            general_data_ratio=0.5,
            class_weighted_loss=True,
            early_stopping_patience=3,
            auto_benchmark=True,
            general_benchmark_path=str(tmp_path / "general_heldout.jsonl"),
            auto_onnx_export=True,
        )

        result = finetune_nli(train_file, eval_path=eval_file, config=cfg)

        assert result.output_dir == cfg.output_dir
        assert result.epochs_completed == 2
        assert result.train_samples == 3
        assert result.eval_samples == 1
        assert result.final_loss == 0.123
        assert result.best_balanced_accuracy == 0.91
        assert result.eval_metrics == {
            "eval_balanced_accuracy": 0.91,
            "eval_f1": 0.88,
        }
        assert result.regression_report == {
            "recommendation": "deploy",
            "regression_acceptable": True,
            "general_accuracy": 0.81,
            "domain_accuracy": 0.93,
            "regression_pp": -0.2,
            "reason": "",
            "domain_set_selected_checkpoint": True,
        }
        assert benchmark_calls == [
            (cfg.output_dir, cfg.general_benchmark_path, eval_file),
        ]
        provenance = TrainingProvenance.read(tmp_path / "model" / PROVENANCE_FILENAME)
        assert len(provenance.content_keys) == 3
        assert result.onnx_path == str(tmp_path / "model" / "onnx")
        assert result.mixed_general_samples == 1
        assert recorder["cache_cleared"] is True
        assert (
            recorder["tokenizer_from_pretrained"][1]["revision"] == "resolved-revision"
        )
        assert recorder["model_from_pretrained"][1]["num_labels"] == 2
        assert recorder["training_args"]["eval_strategy"] == "steps"
        assert recorder["training_args"]["load_best_model_at_end"] is True
        assert recorder["early_stopping"] == {"early_stopping_patience": 3}
        assert (
            recorder["trainer_kwargs"]["compute_metrics"]
            is finetune_mod._compute_metrics
        )
        assert recorder["trainer_kwargs"]["callbacks"]
        assert recorder["model_saved"] == cfg.output_dir
        assert recorder["tokenizer_saved"] == cfg.output_dir
        assert recorder["onnx"] == (cfg.output_dir, result.onnx_path)
        assert recorder["model"].to_calls == ["cpu"]
        assert recorder["released_cuda"] is True

    def test_finetune_without_eval_uses_no_eval_training_arguments(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without evaluation data the trainer runs without evaluation."""
        import director_ai.core._device as device_mod
        import director_ai.core.scoring.nli as nli_mod
        from director_ai.core.training import finetune as finetune_mod
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        recorder: dict[str, Any] = {}
        self._install_fake_transformers(monkeypatch, recorder)
        train_file = self._write_rows(
            tmp_path / "train.jsonl",
            [{"premise": "P", "hypothesis": "H", "label": 1}],
        )
        monkeypatch.setattr(
            finetune_mod,
            "resolve_model_revision",
            lambda model, revision: None,
        )
        monkeypatch.setattr(nli_mod, "clear_model_cache", lambda: None)
        monkeypatch.setattr(device_mod, "release_torch_cuda", lambda: None)
        monkeypatch.setattr(
            finetune_mod,
            "_prepare_dataset",
            lambda rows, tokenizer, max_length, is_factcg: list(rows),
        )

        cfg = FinetuneConfig(output_dir=str(tmp_path / "model"), save_strategy="epoch")

        result = finetune_nli(train_file, config=cfg)

        assert result.eval_samples == 0
        assert result.best_balanced_accuracy == 0.0
        assert recorder["training_args"]["eval_strategy"] == "no"
        assert recorder["training_args"]["eval_steps"] is None
        assert recorder["training_args"]["save_strategy"] == "epoch"
        assert recorder["training_args"]["save_steps"] == 0
        assert recorder["training_args"]["load_best_model_at_end"] is False
        assert recorder["training_args"]["metric_for_best_model"] is None
        assert recorder["trainer_kwargs"]["eval_dataset"] is None
        assert recorder["trainer_kwargs"]["compute_metrics"] is None
        assert recorder["trainer_kwargs"]["callbacks"] is None

    def test_finetune_auto_benchmark_failure_rejects(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing auto-benchmark yields a reject report."""
        import director_ai.core._device as device_mod
        import director_ai.core.scoring.nli as nli_mod
        import director_ai.core.training.finetune_benchmark as benchmark_mod
        from director_ai.core.training import finetune as finetune_mod
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        recorder: dict[str, Any] = {}
        self._install_fake_transformers(monkeypatch, recorder)
        train_file = self._write_rows(
            tmp_path / "train.jsonl",
            [{"premise": "P", "hypothesis": "H", "label": 1}],
        )
        monkeypatch.setattr(
            finetune_mod,
            "resolve_model_revision",
            lambda model, revision: None,
        )
        monkeypatch.setattr(nli_mod, "clear_model_cache", lambda: None)
        monkeypatch.setattr(device_mod, "release_torch_cuda", lambda: None)
        monkeypatch.setattr(
            finetune_mod,
            "_prepare_dataset",
            lambda rows, tokenizer, max_length, is_factcg: list(rows),
        )

        def fail_benchmark(
            output_dir: str, general_path: str | None, eval_path: Path | None
        ) -> SimpleNamespace:
            raise RuntimeError("benchmark unavailable")

        monkeypatch.setattr(benchmark_mod, "benchmark_finetuned_model", fail_benchmark)

        result = finetune_nli(
            train_file,
            config=FinetuneConfig(
                output_dir=str(tmp_path / "model"),
                auto_benchmark=True,
            ),
        )

        assert result.regression_report == {
            "recommendation": "reject",
            "regression_acceptable": False,
            "general_accuracy": 0.0,
            "domain_accuracy": 0.0,
            "regression_pp": 0.0,
            "reason": "auto-benchmark failed",
            "domain_set_selected_checkpoint": False,
        }

    def test_finetune_cleanup_tolerates_missing_trainer_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cleanup tolerates a trainer without a model."""
        import director_ai.core._device as device_mod
        import director_ai.core.scoring.nli as nli_mod
        from director_ai.core.training import finetune as finetune_mod
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        recorder: dict[str, Any] = {"drop_model_on_train": True}
        self._install_fake_transformers(monkeypatch, recorder)
        train_file = self._write_rows(
            tmp_path / "train.jsonl",
            [{"premise": "P", "hypothesis": "H", "label": 1}],
        )
        monkeypatch.setattr(
            finetune_mod,
            "resolve_model_revision",
            lambda model, revision: None,
        )
        monkeypatch.setattr(nli_mod, "clear_model_cache", lambda: None)
        monkeypatch.setattr(
            device_mod,
            "release_torch_cuda",
            lambda: recorder.setdefault("released_cuda", True),
        )
        monkeypatch.setattr(
            finetune_mod,
            "_prepare_dataset",
            lambda rows, tokenizer, max_length, is_factcg: list(rows),
        )

        result = finetune_nli(
            train_file,
            config=FinetuneConfig(output_dir=str(tmp_path / "model")),
        )

        assert result.final_loss == 0.123
        assert recorder["model"].to_calls == []
        assert recorder["released_cuda"] is True

    def test_finetune_refuses_evaluation_rows_shared_with_training(
        self, tmp_path: Path
    ) -> None:
        """Evaluation rows sharing content or a cluster with training are refused."""
        from director_ai.core.training.finetune import finetune_nli

        train_file = self._write_rows(
            tmp_path / "train.jsonl",
            [
                {
                    "premise": "P1",
                    "hypothesis": "H1",
                    "label": 1,
                    "source_cluster": "doc-1",
                },
                {"premise": "P2", "hypothesis": "H2", "label": 0},
            ],
        )
        eval_file = self._write_rows(
            tmp_path / "eval.jsonl",
            [
                {
                    "premise": "Other",
                    "hypothesis": "Claim",
                    "label": 1,
                    "source_cluster": "doc-1",
                },
                {"premise": "p2", "hypothesis": " h2 ", "label": 0},
            ],
        )

        with pytest.raises(ValueError, match="2 evaluation rows share"):
            finetune_nli(train_file, eval_path=eval_file)

    def test_finetune_auto_benchmark_without_general_benchmark_rejects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The auto-benchmark rejects when no general benchmark is configured."""
        import director_ai.core._device as device_mod
        import director_ai.core.scoring.nli as nli_mod
        from director_ai.core.training import finetune as finetune_mod
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        recorder: dict[str, Any] = {}
        self._install_fake_transformers(monkeypatch, recorder)
        train_file = self._write_rows(
            tmp_path / "train.jsonl",
            [{"premise": "P", "hypothesis": "H", "label": 1}],
        )
        monkeypatch.setattr(
            finetune_mod, "resolve_model_revision", lambda model, revision: None
        )
        monkeypatch.setattr(nli_mod, "clear_model_cache", lambda: None)
        monkeypatch.setattr(device_mod, "release_torch_cuda", lambda: None)
        monkeypatch.setattr(
            finetune_mod,
            "_prepare_dataset",
            lambda rows, tokenizer, max_length, is_factcg: list(rows),
        )

        result = finetune_nli(
            train_file,
            config=FinetuneConfig(
                output_dir=str(tmp_path / "model"), auto_benchmark=True
            ),
        )

        assert result.regression_report["recommendation"] == "reject"
        assert result.regression_report["regression_acceptable"] is False
        assert result.regression_report["reason"] == "no general benchmark supplied"


class TestExports:
    """Public exports of the fine-tuning API."""

    def test_finetune_in_core_all(self) -> None:
        """The fine-tuning names are in the core exports."""
        from director_ai.core import __all__

        assert "finetune_nli" in __all__
        assert "FinetuneConfig" in __all__
        assert "FinetuneResult" in __all__

    def test_importable(self) -> None:
        """The fine-tuning API imports from the core package."""
        from director_ai.core import FinetuneConfig, FinetuneResult, finetune_nli

        assert callable(finetune_nli)
        assert FinetuneConfig is not None
        assert FinetuneResult is not None


class TestCliFinetune:
    """The finetune CLI entry point."""

    def test_finetune_no_args_exits(self) -> None:
        """The CLI exits without a training file."""
        from director_ai.cli import main

        with pytest.raises(SystemExit):
            main(["finetune"])

    def test_finetune_missing_file_exits(self) -> None:
        """The CLI exits for a missing training file."""
        from director_ai.cli import main

        with pytest.raises(SystemExit):
            main(["finetune", "/nonexistent_finetune_test_xyz/train.jsonl"])
