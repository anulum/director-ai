# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — GPU-backed Fine-tuning Integration Tests

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_SMALL_MODEL = "microsoft/deberta-v3-base"
_FACTCG_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
_NON_FACTCG_MODEL = "microsoft/deberta-v3-base"


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _make_nli_data(n_pos=50, n_neg=50):
    rows = []
    for i in range(n_pos):
        rows.append(
            {
                "premise": f"The capital of country {i} is city {i}.",
                "hypothesis": f"City {i} is a capital.",
                "label": 1,
            },
        )
    for i in range(n_neg):
        rows.append(
            {
                "premise": f"Country {i} has no coastline.",
                "hypothesis": f"Country {i} is an island.",
                "label": 0,
            },
        )
    return rows


# â”€â”€ _evaluate_model (real inference) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestEvaluateModelReal:
    """Test _evaluate_model with real FactCG model on CPU (fits in RAM)."""

    @pytest.fixture(scope="class")
    def benchmark_file(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("bench")
        rows = [
            {
                "premise": "Paris is the capital of France.",
                "hypothesis": "Paris is in France.",
                "label": 1,
            },
            {
                "premise": "Paris is the capital of France.",
                "hypothesis": "Paris is in Germany.",
                "label": 0,
            },
            {
                "premise": "Water boils at 100 degrees Celsius.",
                "hypothesis": "Water boils at 100C.",
                "label": 1,
            },
            {
                "premise": "Water boils at 100 degrees Celsius.",
                "hypothesis": "Water freezes at 100C.",
                "label": 0,
            },
            {
                "premise": "The Earth orbits the Sun.",
                "hypothesis": "The Sun orbits the Earth.",
                "label": 0,
            },
            {
                "premise": "The Earth orbits the Sun.",
                "hypothesis": "Earth goes around the Sun.",
                "label": 1,
            },
        ]
        f = tmp / "bench.jsonl"
        _write_jsonl(f, rows)
        return f

    def test_evaluate_returns_metrics(self, benchmark_file):
        from director_ai.core.finetune_benchmark import _evaluate_model

        result = _evaluate_model(
            _FACTCG_MODEL,
            [
                {
                    "premise": "Paris is the capital of France.",
                    "hypothesis": "Paris is in France.",
                    "label": 1,
                },
                {
                    "premise": "Paris is the capital of France.",
                    "hypothesis": "Paris is in Germany.",
                    "label": 0,
                },
            ],
            batch_size=2,
        )
        assert "balanced_accuracy" in result
        assert "f1" in result
        assert 0 <= result["balanced_accuracy"] <= 1

    def test_factcg_gets_easy_cases_right(self):
        from director_ai.core.finetune_benchmark import _evaluate_model

        samples = [
            {
                "premise": "Paris is the capital of France.",
                "hypothesis": "Paris is in France.",
                "label": 1,
            },
            {
                "premise": "Paris is the capital of France.",
                "hypothesis": "Paris is in Germany.",
                "label": 0,
            },
            {
                "premise": "Water boils at 100 degrees Celsius.",
                "hypothesis": "Water boils at 100C.",
                "label": 1,
            },
            {
                "premise": "Water boils at 100 degrees Celsius.",
                "hypothesis": "Water freezes at 100C.",
                "label": 0,
            },
        ]
        result = _evaluate_model(_FACTCG_MODEL, samples, batch_size=4)
        assert result["balanced_accuracy"] >= 0.75

    def test_benchmark_end_to_end(self, benchmark_file):
        from director_ai.core.finetune_benchmark import benchmark_finetuned_model

        report = benchmark_finetuned_model(
            _FACTCG_MODEL,
            general_path=benchmark_file,
        )
        assert report.general_accuracy > 0
        assert report.recommendation in ("deploy", "deploy_domain_only", "reject")
        assert report.regression_pp != 0.0


# â”€â”€ _prepare_dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestPrepareDataset:
    def test_factcg_template(self):
        from director_ai.core.finetune import _prepare_dataset

        tok = transformers.AutoTokenizer.from_pretrained(_SMALL_MODEL)
        rows = _make_nli_data(10, 10)
        ds = _prepare_dataset(rows, tok, max_length=128, is_factcg=True)
        assert len(ds) == 20
        assert "input_ids" in ds.column_names
        assert "labels" in ds.column_names

    def test_pair_template(self):
        from director_ai.core.finetune import _prepare_dataset

        tok = transformers.AutoTokenizer.from_pretrained(_SMALL_MODEL)
        rows = _make_nli_data(10, 10)
        ds = _prepare_dataset(rows, tok, max_length=128, is_factcg=False)
        assert len(ds) == 20
        assert "input_ids" in ds.column_names


# â”€â”€ finetune_nli (micro-training on GPU) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestFinetuneNliGPU:
    """Micro-training: 1 epoch, bs=2, 100 samples, deberta-v3-base."""

    def test_finetune_end_to_end(self, tmp_path):
        from director_ai.core.finetune import FinetuneConfig, finetune_nli

        train_rows = _make_nli_data(60, 60)
        eval_rows = _make_nli_data(10, 10)
        train_f = tmp_path / "train.jsonl"
        eval_f = tmp_path / "eval.jsonl"
        _write_jsonl(train_f, train_rows)
        _write_jsonl(eval_f, eval_rows)

        cfg = FinetuneConfig(
            base_model=_SMALL_MODEL,
            output_dir=str(tmp_path / "model_out"),
            epochs=1,
            batch_size=4,
            learning_rate=5e-5,
            fp16=True,
            max_length=128,
            eval_steps=20,
            save_strategy="steps",
        )
        result = finetune_nli(str(train_f), eval_path=str(eval_f), config=cfg)

        assert result.output_dir == str(tmp_path / "model_out")
        assert result.epochs_completed == 1
        assert result.train_samples == 120
        assert result.eval_samples == 20
        assert result.final_loss > 0
        assert result.best_balanced_accuracy > 0
        assert (tmp_path / "model_out" / "config.json").exists()

    def test_finetune_class_weighted(self, tmp_path):
        from director_ai.core.finetune import FinetuneConfig, finetune_nli

        # Imbalanced: 90 pos, 30 neg
        train_rows = _make_nli_data(90, 30)
        train_f = tmp_path / "train.jsonl"
        _write_jsonl(train_f, train_rows)

        cfg = FinetuneConfig(
            base_model=_SMALL_MODEL,
            output_dir=str(tmp_path / "weighted_out"),
            epochs=1,
            batch_size=4,
            fp16=True,
            max_length=128,
            class_weighted_loss=True,
        )
        result = finetune_nli(str(train_f), config=cfg)
        assert result.train_samples == 120
        assert result.final_loss > 0

    def test_finetune_mix_general_data(self, tmp_path):
        from director_ai.core.finetune import FinetuneConfig, finetune_nli

        train_rows = _make_nli_data(60, 60)
        general_rows = _make_nli_data(30, 30)
        train_f = tmp_path / "train.jsonl"
        general_f = tmp_path / "general.jsonl"
        _write_jsonl(train_f, train_rows)
        _write_jsonl(general_f, general_rows)

        cfg = FinetuneConfig(
            base_model=_SMALL_MODEL,
            output_dir=str(tmp_path / "mixed_out"),
            epochs=1,
            batch_size=4,
            fp16=True,
            max_length=128,
            mix_general_data=True,
            general_data_path=str(general_f),
            general_data_ratio=0.2,
        )
        result = finetune_nli(str(train_f), config=cfg)
        assert result.mixed_general_samples > 0

    def test_finetune_early_stopping(self, tmp_path):
        from director_ai.core.finetune import FinetuneConfig, finetune_nli

        train_rows = _make_nli_data(60, 60)
        eval_rows = _make_nli_data(10, 10)
        train_f = tmp_path / "train.jsonl"
        eval_f = tmp_path / "eval.jsonl"
        _write_jsonl(train_f, train_rows)
        _write_jsonl(eval_f, eval_rows)

        cfg = FinetuneConfig(
            base_model=_SMALL_MODEL,
            output_dir=str(tmp_path / "es_out"),
            epochs=5,
            batch_size=4,
            fp16=True,
            max_length=128,
            eval_steps=10,
            early_stopping_patience=2,
        )
        result = finetune_nli(str(train_f), eval_path=str(eval_f), config=cfg)
        assert result.best_balanced_accuracy > 0

    def test_finetune_empty_raises(self, tmp_path):
        from director_ai.core.finetune import finetune_nli

        empty_f = tmp_path / "empty.jsonl"
        empty_f.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No valid samples"):
            finetune_nli(str(empty_f))

    def test_finetune_auto_benchmark(self, tmp_path):
        from director_ai.core.finetune import FinetuneConfig, finetune_nli

        train_rows = _make_nli_data(60, 60)
        eval_rows = _make_nli_data(10, 10)
        train_f = tmp_path / "train.jsonl"
        eval_f = tmp_path / "eval.jsonl"
        _write_jsonl(train_f, train_rows)
        _write_jsonl(eval_f, eval_rows)

        cfg = FinetuneConfig(
            base_model=_SMALL_MODEL,
            output_dir=str(tmp_path / "bench_out"),
            epochs=1,
            batch_size=4,
            fp16=True,
            max_length=128,
            eval_steps=20,
            auto_benchmark=True,
        )
        result = finetune_nli(str(train_f), eval_path=str(eval_f), config=cfg)
        assert result.output_dir == str(tmp_path / "bench_out")

    def test_finetune_auto_onnx_export(self, tmp_path):
        from director_ai.core.finetune import FinetuneConfig, finetune_nli

        train_rows = _make_nli_data(60, 60)
        train_f = tmp_path / "train.jsonl"
        _write_jsonl(train_f, train_rows)

        cfg = FinetuneConfig(
            base_model=_SMALL_MODEL,
            output_dir=str(tmp_path / "onnx_out"),
            epochs=1,
            batch_size=4,
            fp16=True,
            max_length=128,
            auto_onnx_export=True,
        )
        result = finetune_nli(str(train_f), config=cfg)
        assert result.output_dir == str(tmp_path / "onnx_out")


# â”€â”€ _evaluate_model non-FactCG branch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestEvaluateNonFactCG:
    """Test _evaluate_model with a non-FactCG model (sep_token template)."""

    def test_non_factcg_uses_sep_template(self):
        from director_ai.core.finetune_benchmark import _evaluate_model

        samples = [
            {
                "premise": "Paris is the capital of France.",
                "hypothesis": "Paris is in France.",
                "label": 1,
            },
            {
                "premise": "Paris is the capital of France.",
                "hypothesis": "Paris is in Germany.",
                "label": 0,
            },
        ]
        result = _evaluate_model(_NON_FACTCG_MODEL, samples, batch_size=2)
        assert "balanced_accuracy" in result
        assert 0 <= result["balanced_accuracy"] <= 1
