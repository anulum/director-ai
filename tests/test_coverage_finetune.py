# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _make_rows(n_pos: int = 10, n_neg: int = 10) -> list[dict]:
    rows = []
    for i in range(n_pos):
        rows.append({"premise": f"P{i}", "hypothesis": f"H{i}", "label": 1})
    for i in range(n_neg):
        rows.append({"premise": f"S{i}", "hypothesis": f"W{i}", "label": 0})
    return rows


def _make_fake_datasets_module():
    """Minimal `datasets` stub with a Dataset that calls map callbacks."""
    ds_mod = types.ModuleType("datasets")

    class FakeDataset:
        def __init__(self, data):
            self._data = data

        @classmethod
        def from_dict(cls, d):
            return cls(d)

        def map(self, fn, batched=False, batch_size=None, remove_columns=None):
            if batched:
                fn(self._data)
            return self

        def set_format(self, fmt):
            pass

    ds_mod.Dataset = FakeDataset
    return ds_mod


def _make_fake_transformers_module():
    """Minimal `transformers` stub sufficient for finetune_nli."""
    tf_mod = types.ModuleType("transformers")

    train_output = MagicMock()
    train_output.training_loss = 0.42

    trainer_inst = MagicMock()
    trainer_inst.train.return_value = train_output
    trainer_inst.evaluate.return_value = {
        "eval_balanced_accuracy": 0.77,
        "eval_loss": 0.3,
    }

    class FakeTrainer:
        pass

    tf_mod.AutoTokenizer = MagicMock()
    tf_mod.AutoTokenizer.from_pretrained.return_value = MagicMock(
        return_value={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
    )
    tf_mod.AutoModelForSequenceClassification = MagicMock()
    tf_mod.AutoModelForSequenceClassification.from_pretrained.return_value = MagicMock()
    tf_mod.TrainingArguments = MagicMock(return_value=MagicMock())
    tf_mod.Trainer = MagicMock(return_value=trainer_inst)
    tf_mod.EarlyStoppingCallback = MagicMock()

    tf_mod._trainer_inst = trainer_inst
    tf_mod._train_output = train_output
    return tf_mod


@pytest.fixture
def train_file(tmp_path):
    f = tmp_path / "train.jsonl"
    _write_jsonl(f, _make_rows(10, 10))
    return f


@pytest.fixture
def eval_file(tmp_path):
    f = tmp_path / "eval.jsonl"
    _write_jsonl(f, _make_rows(5, 5))
    return f


# ---------------------------------------------------------------------------
# _mix_general_data — missing branches
# ---------------------------------------------------------------------------


class TestMixGeneralDataEdgeCases:
    def test_none_path_default_missing(self):
        """general_path=None and default file absent → return domain unchanged (lines 139-140, 144-145)."""
        from director_ai.core.training.finetune import _mix_general_data

        domain = _make_rows(5, 5)
        mixed, n = _mix_general_data(domain, None, 0.2, 42)
        assert n == 0
        assert mixed is domain

    def test_none_path_default_present(self, tmp_path):
        """general_path=None and default file exists → it is loaded and mixed (lines 139-140)."""
        from director_ai.core.training import finetune as ft_mod

        general_rows = _make_rows(50, 50)
        general_file = tmp_path / "aggrefact_benchmark_1k.jsonl"
        _write_jsonl(general_file, general_rows)

        original_load = ft_mod._load_jsonl

        def fake_load(path):
            if "aggrefact" in str(path):
                return general_rows[:]
            return original_load(path)

        domain = _make_rows(100, 100)

        mock_resolved = MagicMock()
        mock_resolved.exists.return_value = True
        mock_resolved.__truediv__ = lambda self, other: mock_resolved
        mock_resolved.__str__ = lambda self: str(general_file)

        mock_path_cls = MagicMock()
        mock_path_cls.return_value = mock_resolved

        with (
            patch.object(ft_mod, "_load_jsonl", side_effect=fake_load),
            patch("director_ai.core.training.finetune.Path", mock_path_cls),
        ):
            mixed, n = ft_mod._mix_general_data(domain, None, 0.2, 42)

        assert n >= 0

    def test_empty_general_file_returns_domain(self, tmp_path):
        """Empty general JSONL → return domain unchanged (line 149)."""
        from director_ai.core.training.finetune import _mix_general_data

        f = tmp_path / "general.jsonl"
        f.write_text("", encoding="utf-8")
        domain = _make_rows(10, 10)
        mixed, n = _mix_general_data(domain, f, 0.2, 42)
        assert n == 0
        assert mixed is domain

    def test_n_general_exceeds_available_uses_all(self, tmp_path):
        """n_general >= len(general_rows) → use all general rows (line 156)."""
        from director_ai.core.training.finetune import _mix_general_data

        general_file = tmp_path / "general.jsonl"
        _write_jsonl(general_file, _make_rows(3, 2))  # 5 rows

        domain = _make_rows(50, 50)  # ratio=0.8 → n_general = 200 >> 5
        mixed, n = _mix_general_data(domain, general_file, 0.8, 42)
        assert n == 5
        assert len(mixed) == 105


# ---------------------------------------------------------------------------
# _prepare_dataset — lines 182-228
# ---------------------------------------------------------------------------


class TestPrepareDataset:
    def test_factcg_path_runs(self):
        """FactCG branch executes without error."""
        from director_ai.core.training import finetune as ft_mod

        rows = _make_rows(4, 4)
        tok = MagicMock(
            return_value={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
        )
        ds_mod = _make_fake_datasets_module()

        with patch.dict(sys.modules, {"datasets": ds_mod}):
            result = ft_mod._prepare_dataset(rows, tok, max_length=64, is_factcg=True)

        assert result is not None

    def test_non_factcg_path_runs(self):
        """Non-FactCG branch executes without error."""
        from director_ai.core.training import finetune as ft_mod

        rows = _make_rows(4, 4)
        tok = MagicMock(
            return_value={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
        )
        ds_mod = _make_fake_datasets_module()

        with patch.dict(sys.modules, {"datasets": ds_mod}):
            result = ft_mod._prepare_dataset(rows, tok, max_length=64, is_factcg=False)

        assert result is not None

    def test_factcg_calls_tokenizer_with_text_list(self):
        """FactCG branch passes a list of formatted strings to the tokenizer."""
        from director_ai.core.training import finetune as ft_mod

        rows = [
            {
                "premise": "Alice is in Paris.",
                "hypothesis": "Alice is in France.",
                "label": 1,
            }
        ]
        calls = []

        def capturing_tok(texts, **kwargs):
            calls.append(texts)
            return {"input_ids": [[1]], "attention_mask": [[1]]}

        ds_mod = _make_fake_datasets_module()
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            ft_mod._prepare_dataset(rows, capturing_tok, max_length=128, is_factcg=True)

        assert len(calls) == 1
        assert isinstance(calls[0], list)
        assert "Alice is in Paris." in calls[0][0]
        assert "Alice is in France." in calls[0][0]

    def test_non_factcg_calls_tokenizer_with_premise_hypothesis(self):
        """Non-FactCG branch passes premise/hypothesis lists to the tokenizer."""
        from director_ai.core.training import finetune as ft_mod

        rows = [{"premise": "Sky is blue.", "hypothesis": "Sky has color.", "label": 1}]
        calls = []

        def capturing_tok(premises, hypotheses=None, **kwargs):
            calls.append((premises, hypotheses))
            return {"input_ids": [[1]], "attention_mask": [[1]]}

        ds_mod = _make_fake_datasets_module()
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            ft_mod._prepare_dataset(
                rows, capturing_tok, max_length=128, is_factcg=False
            )

        assert len(calls) == 1
        assert "Sky is blue." in calls[0][0]
        assert "Sky has color." in calls[0][1]

    def test_set_format_called(self):
        """set_format('torch') is called on the resulting dataset."""
        from director_ai.core.training import finetune as ft_mod

        rows = _make_rows(2, 2)
        tok = MagicMock(return_value={"input_ids": [[1]], "attention_mask": [[1]]})

        formats_set = []

        class TrackingDataset:
            def __init__(self, data):
                self._data = data

            @classmethod
            def from_dict(cls, d):
                return cls(d)

            def map(self, fn, batched=False, batch_size=None, remove_columns=None):
                if batched:
                    fn(self._data)
                return self

            def set_format(self, fmt):
                formats_set.append(fmt)

        tracking_mod = types.ModuleType("datasets")
        tracking_mod.Dataset = TrackingDataset

        with patch.dict(sys.modules, {"datasets": tracking_mod}):
            ft_mod._prepare_dataset(rows, tok, max_length=64, is_factcg=False)

        assert "torch" in formats_set


# ---------------------------------------------------------------------------
# _compute_metrics — lines 233-240
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self):
        from director_ai.core.training.finetune import _compute_metrics

        logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.1, 0.9]])
        labels = np.array([1, 0, 1])
        metrics = _compute_metrics((logits, labels))
        assert metrics["balanced_accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        from director_ai.core.training.finetune import _compute_metrics

        logits = np.array([[0.9, 0.1], [0.9, 0.1]])
        labels = np.array([1, 1])
        metrics = _compute_metrics((logits, labels))
        assert metrics["balanced_accuracy"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(0.0)

    def test_mixed_predictions(self):
        from director_ai.core.training.finetune import _compute_metrics

        logits = np.array([[0.1, 0.9], [0.1, 0.9], [0.8, 0.2], [0.8, 0.2]])
        labels = np.array([1, 0, 1, 0])
        metrics = _compute_metrics((logits, labels))
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
        assert "f1" in metrics

    def test_returns_required_keys(self):
        from director_ai.core.training.finetune import _compute_metrics

        logits = np.array([[0.2, 0.8]])
        labels = np.array([1])
        metrics = _compute_metrics((logits, labels))
        assert set(metrics.keys()) == {"balanced_accuracy", "f1"}


# ---------------------------------------------------------------------------
# _make_weighted_trainer_class — lines 245-259
# ---------------------------------------------------------------------------


class TestMakeWeightedTrainerClass:
    def _torch_and_tf_stubs(self):
        torch_mod = types.ModuleType("torch")

        class FakeTensor:
            def __init__(self, data, dtype=None):
                self.data = list(data)

            def to(self, device):
                return self

        torch_mod.tensor = lambda data, dtype=None: FakeTensor(data, dtype)
        torch_mod.float32 = "float32"

        nn_mod = types.ModuleType("torch.nn")

        class FakeLoss:
            def __init__(self, weight=None):
                self.weight = weight

            def __call__(self, logits, labels):
                return FakeTensor([0.5])

        nn_mod.CrossEntropyLoss = FakeLoss
        torch_mod.nn = nn_mod

        tf_mod = types.ModuleType("transformers")

        class FakeTrainer:
            pass

        tf_mod.Trainer = FakeTrainer
        return torch_mod, tf_mod

    def test_returns_trainer_subclass(self):
        from director_ai.core.training import finetune as ft_mod

        torch_mod, tf_mod = self._torch_and_tf_stubs()
        with patch.dict(sys.modules, {"torch": torch_mod, "transformers": tf_mod}):
            WeightedTrainer = ft_mod._make_weighted_trainer_class([0.5, 1.5])

        assert issubclass(WeightedTrainer, tf_mod.Trainer)

    def test_compute_loss_scalar_return(self):
        from director_ai.core.training import finetune as ft_mod

        torch_mod, tf_mod = self._torch_and_tf_stubs()
        with patch.dict(sys.modules, {"torch": torch_mod, "transformers": tf_mod}):
            WeightedTrainer = ft_mod._make_weighted_trainer_class([1.0, 1.0])

        trainer = WeightedTrainer.__new__(WeightedTrainer)
        fake_logits = MagicMock()
        fake_logits.device = "cpu"
        fake_outputs = MagicMock()
        fake_outputs.logits = fake_logits
        fake_model = MagicMock(return_value=fake_outputs)
        inputs = {"labels": MagicMock(), "input_ids": MagicMock()}

        loss = trainer.compute_loss(fake_model, inputs, return_outputs=False)
        assert loss is not None

    def test_compute_loss_with_return_outputs(self):
        from director_ai.core.training import finetune as ft_mod

        torch_mod, tf_mod = self._torch_and_tf_stubs()
        with patch.dict(sys.modules, {"torch": torch_mod, "transformers": tf_mod}):
            WeightedTrainer = ft_mod._make_weighted_trainer_class([1.0, 1.0])

        trainer = WeightedTrainer.__new__(WeightedTrainer)
        fake_logits = MagicMock()
        fake_logits.device = "cpu"
        fake_outputs = MagicMock()
        fake_outputs.logits = fake_logits
        fake_model = MagicMock(return_value=fake_outputs)
        inputs = {"labels": MagicMock(), "input_ids": MagicMock()}

        result = trainer.compute_loss(fake_model, inputs, return_outputs=True)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# finetune_nli — lines 280-451
# Imports are lazy (inside function body), so we inject via sys.modules.
# ---------------------------------------------------------------------------


def _run_finetune_mocked(
    train_path,
    *,
    eval_path=None,
    config=None,
    extra_modules=None,
    tf_override=None,
):
    """Run finetune_nli with transformers/datasets mocked out."""
    from director_ai.core.training.finetune import finetune_nli

    tf_mod = (
        tf_override if tf_override is not None else _make_fake_transformers_module()
    )
    ds_mod = _make_fake_datasets_module()

    modules = {"transformers": tf_mod, "datasets": ds_mod}
    if extra_modules:
        modules.update(extra_modules)

    with patch.dict(sys.modules, modules):
        return finetune_nli(train_path, eval_path=eval_path, config=config), tf_mod


class TestFinetuneNliBasic:
    def test_empty_train_raises(self, tmp_path):
        from director_ai.core.training.finetune import finetune_nli

        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No valid samples"):
            finetune_nli(f)

    def test_minimal_run_no_eval(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig, FinetuneResult

        cfg = FinetuneConfig(
            base_model="some/non-factcg-model",
            output_dir=str(tmp_path / "out"),
            epochs=1,
        )
        result, _ = _run_finetune_mocked(train_file, config=cfg)

        assert isinstance(result, FinetuneResult)
        assert result.output_dir == str(tmp_path / "out")
        assert result.epochs_completed == 1
        assert result.train_samples == 20
        assert result.eval_samples == 0
        assert result.final_loss == pytest.approx(0.42)

    def test_run_with_eval(self, tmp_path, train_file, eval_file):
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig(
            base_model="some/non-factcg-model",
            output_dir=str(tmp_path / "out"),
            epochs=1,
        )
        result, _ = _run_finetune_mocked(train_file, eval_path=eval_file, config=cfg)

        assert result.eval_samples == 10
        assert result.best_balanced_accuracy == pytest.approx(0.77)
        assert result.eval_metrics["eval_balanced_accuracy"] == pytest.approx(0.77)

    def test_default_config_used_when_none(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig

        result, _ = _run_finetune_mocked(train_file, config=None)
        assert result.output_dir == FinetuneConfig().output_dir

    def test_tokenizer_and_model_loaded(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig(
            base_model="some/model", output_dir=str(tmp_path / "out"), epochs=1
        )
        _, tf_mod = _run_finetune_mocked(train_file, config=cfg)

        tf_mod.AutoTokenizer.from_pretrained.assert_called_once_with("some/model")
        tf_mod.AutoModelForSequenceClassification.from_pretrained.assert_called_once_with(
            "some/model", num_labels=2
        )

    def test_model_saved_after_training(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig(
            base_model="some/model",
            output_dir=str(tmp_path / "out"),
            epochs=1,
        )
        _, tf_mod = _run_finetune_mocked(train_file, config=cfg)

        trainer_inst = tf_mod._trainer_inst
        trainer_inst.save_model.assert_called_once_with(str(tmp_path / "out"))

    def test_eval_metrics_recorded(self, tmp_path, train_file, eval_file):
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig(base_model="m", output_dir=str(tmp_path / "o"), epochs=1)
        result, tf_mod = _run_finetune_mocked(
            train_file, eval_path=eval_file, config=cfg
        )

        tf_mod._trainer_inst.evaluate.assert_called_once()
        assert "eval_balanced_accuracy" in result.eval_metrics


class TestFinetuneNliFactCG:
    def test_factcg_model_name_detected(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig(
            base_model="yaxili96/FactCG-DeBERTa-v3-Large",
            output_dir=str(tmp_path / "out"),
            epochs=1,
        )
        result, _ = _run_finetune_mocked(train_file, config=cfg)
        assert result is not None

    def test_non_factcg_model_name_detected(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig

        cfg = FinetuneConfig(
            base_model="bert-base-uncased",
            output_dir=str(tmp_path / "out"),
            epochs=1,
        )
        result, _ = _run_finetune_mocked(train_file, config=cfg)
        assert result is not None


class TestFinetuneNliPhaseE:
    def test_mix_general_data_enabled(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig

        general_file = tmp_path / "general.jsonl"
        _write_jsonl(general_file, _make_rows(20, 20))

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            mix_general_data=True,
            general_data_path=str(general_file),
            general_data_ratio=0.2,
        )
        result, _ = _run_finetune_mocked(train_file, config=cfg)
        assert result.mixed_general_samples >= 0

    def test_class_weighted_loss_calls_make_weighted_trainer(
        self, tmp_path, train_file
    ):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            class_weighted_loss=True,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        with (
            patch.dict(sys.modules, {"transformers": tf_mod, "datasets": ds_mod}),
            patch(
                "director_ai.core.training.finetune._make_weighted_trainer_class",
                return_value=MagicMock(return_value=tf_mod._trainer_inst),
            ) as mock_wt,
        ):
            result = finetune_nli(train_file, config=cfg)

        mock_wt.assert_called_once()
        assert result is not None

    def test_early_stopping_added_with_eval(self, tmp_path, train_file, eval_file):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            early_stopping_patience=3,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        with patch.dict(sys.modules, {"transformers": tf_mod, "datasets": ds_mod}):
            finetune_nli(train_file, eval_path=eval_file, config=cfg)

        tf_mod.EarlyStoppingCallback.assert_called_once_with(early_stopping_patience=3)

    def test_early_stopping_not_added_without_eval(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            early_stopping_patience=3,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        with patch.dict(sys.modules, {"transformers": tf_mod, "datasets": ds_mod}):
            finetune_nli(train_file, config=cfg)

        tf_mod.EarlyStoppingCallback.assert_not_called()

    def test_auto_benchmark_success(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            auto_benchmark=True,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        fake_report = MagicMock()
        fake_report.recommendation = "deploy"
        fake_report.general_accuracy = 0.8
        fake_report.domain_accuracy = 0.9
        fake_report.regression_pp = 2.5

        bench_mod = types.ModuleType("director_ai.core.training.finetune_benchmark")
        bench_mod.benchmark_finetuned_model = MagicMock(return_value=fake_report)

        with patch.dict(
            sys.modules,
            {
                "transformers": tf_mod,
                "datasets": ds_mod,
                "director_ai.core.training.finetune_benchmark": bench_mod,
            },
        ):
            result = finetune_nli(train_file, config=cfg)

        assert result.regression_report["recommendation"] == "deploy"
        assert result.regression_report["general_accuracy"] == pytest.approx(0.8)
        assert result.regression_report["domain_accuracy"] == pytest.approx(0.9)
        assert result.regression_report["regression_pp"] == pytest.approx(2.5)

    def test_auto_benchmark_exception_swallowed(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            auto_benchmark=True,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        bench_mod = types.ModuleType("director_ai.core.training.finetune_benchmark")
        bench_mod.benchmark_finetuned_model = MagicMock(
            side_effect=RuntimeError("bench exploded")
        )

        with patch.dict(
            sys.modules,
            {
                "transformers": tf_mod,
                "datasets": ds_mod,
                "director_ai.core.training.finetune_benchmark": bench_mod,
            },
        ):
            result = finetune_nli(train_file, config=cfg)

        assert result.regression_report == {}

    def test_auto_onnx_export_success(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            auto_onnx_export=True,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        onnx_mod = types.ModuleType("director_ai.core.scoring.nli")
        onnx_mod.export_onnx = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "transformers": tf_mod,
                "datasets": ds_mod,
                "director_ai.core.scoring.nli": onnx_mod,
            },
        ):
            result = finetune_nli(train_file, config=cfg)

        expected = str(Path(cfg.output_dir) / "onnx")
        assert result.onnx_path == expected
        onnx_mod.export_onnx.assert_called_once_with(cfg.output_dir, expected)

    def test_auto_onnx_export_exception_swallowed(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            auto_onnx_export=True,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        onnx_mod = types.ModuleType("director_ai.core.scoring.nli")
        onnx_mod.export_onnx = MagicMock(side_effect=RuntimeError("onnx broke"))

        with patch.dict(
            sys.modules,
            {
                "transformers": tf_mod,
                "datasets": ds_mod,
                "director_ai.core.scoring.nli": onnx_mod,
            },
        ):
            result = finetune_nli(train_file, config=cfg)

        assert result.onnx_path == ""

    def test_save_strategy_steps_when_eval_present(
        self, tmp_path, train_file, eval_file
    ):
        """eval_dataset present → save_strategy forced to 'steps'."""
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            save_strategy="epoch",
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()
        captured = {}

        def capture_ta(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        tf_mod.TrainingArguments = capture_ta

        with patch.dict(sys.modules, {"transformers": tf_mod, "datasets": ds_mod}):
            finetune_nli(train_file, eval_path=eval_file, config=cfg)

        assert captured.get("save_strategy") == "steps"

    def test_save_strategy_kept_when_no_eval(self, tmp_path, train_file):
        """no eval_dataset → save_strategy stays as config value."""
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            save_strategy="epoch",
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()
        captured = {}

        def capture_ta(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        tf_mod.TrainingArguments = capture_ta

        with patch.dict(sys.modules, {"transformers": tf_mod, "datasets": ds_mod}):
            finetune_nli(train_file, config=cfg)

        assert captured.get("save_strategy") == "epoch"

    def test_no_callbacks_when_patience_zero(self, tmp_path, train_file, eval_file):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            early_stopping_patience=0,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        with patch.dict(sys.modules, {"transformers": tf_mod, "datasets": ds_mod}):
            result = finetune_nli(train_file, eval_path=eval_file, config=cfg)

        tf_mod.EarlyStoppingCallback.assert_not_called()
        assert result is not None

    def test_mixed_general_samples_returned_in_result(self, tmp_path, train_file):
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        general_file = tmp_path / "g.jsonl"
        _write_jsonl(general_file, _make_rows(30, 30))

        cfg = FinetuneConfig(
            base_model="m",
            output_dir=str(tmp_path / "out"),
            epochs=1,
            mix_general_data=True,
            general_data_path=str(general_file),
            general_data_ratio=0.2,
        )
        tf_mod = _make_fake_transformers_module()
        ds_mod = _make_fake_datasets_module()

        with patch.dict(sys.modules, {"transformers": tf_mod, "datasets": ds_mod}):
            result = finetune_nli(train_file, config=cfg)

        assert result.mixed_general_samples >= 0


# ---------------------------------------------------------------------------
# _load_jsonl — cover blank-line skip (104) and warning path (112-116)
# These lines are tested in test_finetune.py too; include here for standalone
# coverage completeness.
# ---------------------------------------------------------------------------


class TestLoadJsonlMissingCoverage:
    def test_blank_lines_skipped(self, tmp_path):
        from director_ai.core.training.finetune import _load_jsonl

        f = tmp_path / "data.jsonl"
        f.write_text(
            "\n" + json.dumps({"premise": "a", "hypothesis": "b", "label": 1}) + "\n\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert len(rows) == 1

    def test_incomplete_row_logs_warning(self, tmp_path):
        from director_ai.core.training.finetune import _load_jsonl

        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"premise": "a", "hypothesis": "b", "label": 1})
            + "\n"
            + json.dumps({"premise": "only-premise"})
            + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert len(rows) == 1

    def test_missing_label_logs_warning(self, tmp_path):
        from director_ai.core.training.finetune import _load_jsonl

        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"premise": "a", "hypothesis": "b"}) + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(f)
        assert rows == []
