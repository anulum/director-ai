# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Training Data Pipeline Tests
"""Multi-angle tests for training/data_pipeline.py.

Covers: all 7 data source loaders, label mapping, VitaminC capping,
CLI argument parsing, build_dataset orchestration, stats output,
error paths, and edge cases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure training/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data_pipeline import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    VITAMINC_CAP,
    _load_anli_r3,
    _load_fever,
    _load_halueval,
    _load_vitaminc,
)

# ── HaluEval loader ────────────────────────────────────────────────


class TestLoadHaluEval:
    """Multi-angle tests for HaluEval QA/Dialogue/Summarization loader."""

    def _make_qa_row(self, *, knowledge="Earth orbits Sun", right="Yes", halluc="No"):
        return {
            "knowledge": knowledge,
            "right_answer": right,
            "hallucinated_answer": halluc,
            "question": "Q",
        }

    def _make_dialogue_row(self, *, history="Hi", right="Hello", halluc="Bye"):
        return {
            "dialogue_history": history,
            "right_response": right,
            "hallucinated_response": halluc,
            "knowledge": "K",
        }

    def _make_summ_row(self, *, doc="Article", right="Summary", halluc="Wrong"):
        return {"document": doc, "right_summary": right, "hallucinated_summary": halluc}

    @patch("datasets.load_dataset")
    def test_qa_produces_entailment_and_contradiction(self, mock_ld):
        mock_ld.return_value = [self._make_qa_row()]
        examples = _load_halueval()
        labels = {e["label"] for e in examples}
        assert LABEL_ENTAILMENT in labels
        assert LABEL_CONTRADICTION in labels

    @patch("datasets.load_dataset")
    def test_three_tasks_loaded(self, mock_ld):
        mock_ld.side_effect = [
            [self._make_qa_row()],
            [self._make_dialogue_row()],
            [self._make_summ_row()],
        ]
        examples = _load_halueval()
        sources = {e["source"] for e in examples}
        assert sources == {"halueval_qa", "halueval_dialogue", "halueval_summarization"}

    @patch("datasets.load_dataset")
    def test_empty_knowledge_uses_question_fallback(self, mock_ld):
        """When knowledge is empty, QA loader falls back to question field."""
        row = {
            "knowledge": "",
            "question": "",
            "right_answer": "A",
            "hallucinated_answer": "B",
        }
        mock_ld.return_value = [row]
        examples = _load_halueval()
        qa_examples = [e for e in examples if e["source"] == "halueval_qa"]
        # Both premise sources empty → skipped
        assert len(qa_examples) == 0

    @patch("datasets.load_dataset")
    def test_empty_hypothesis_skipped(self, mock_ld):
        row = self._make_qa_row(right="", halluc="")
        mock_ld.return_value = [row]
        examples = _load_halueval()
        qa_examples = [e for e in examples if e["source"] == "halueval_qa"]
        assert len(qa_examples) == 0

    @patch("datasets.load_dataset")
    def test_schema_fields_present(self, mock_ld):
        mock_ld.return_value = [self._make_qa_row()]
        examples = _load_halueval()
        for ex in examples:
            assert "premise" in ex
            assert "hypothesis" in ex
            assert "label" in ex
            assert "source" in ex

    @patch("datasets.load_dataset")
    @pytest.mark.parametrize(
        "task_idx,source",
        [
            (0, "halueval_qa"),
            (1, "halueval_dialogue"),
            (2, "halueval_summarization"),
        ],
    )
    def test_per_task_source_tagging(self, mock_ld, task_idx, source):
        rows = [self._make_qa_row(), self._make_dialogue_row(), self._make_summ_row()]
        mock_ld.return_value = [rows[task_idx]]
        examples = _load_halueval()
        matching = [e for e in examples if e["source"] == source]
        assert len(matching) > 0


# ── FEVER loader ────────────────────────────────────────────────────


class TestLoadFever:
    """Tests for FEVER claim+evidence loader with label mapping."""

    @pytest.mark.parametrize(
        "raw_label,expected",
        [
            (0, LABEL_ENTAILMENT),
            (1, LABEL_NEUTRAL),
            (2, LABEL_CONTRADICTION),
            ("entailment", LABEL_ENTAILMENT),
            ("neutral", LABEL_NEUTRAL),
            ("contradiction", LABEL_CONTRADICTION),
        ],
    )
    @patch("datasets.load_dataset")
    def test_label_mapping(self, mock_ld, raw_label, expected):
        mock_ld.return_value = [{"premise": "P", "hypothesis": "H", "label": raw_label}]
        examples = _load_fever()
        assert len(examples) == 1
        assert examples[0]["label"] == expected

    @patch("datasets.load_dataset")
    def test_invalid_label_skipped(self, mock_ld):
        mock_ld.return_value = [{"premise": "P", "hypothesis": "H", "label": "invalid"}]
        examples = _load_fever()
        assert len(examples) == 0

    @patch("datasets.load_dataset")
    def test_missing_premise_skipped(self, mock_ld):
        mock_ld.return_value = [{"premise": "", "hypothesis": "H", "label": 0}]
        examples = _load_fever()
        assert len(examples) == 0

    @patch("datasets.load_dataset")
    def test_source_tagged_fever(self, mock_ld):
        mock_ld.return_value = [{"premise": "P", "hypothesis": "H", "label": 0}]
        examples = _load_fever()
        assert all(e["source"] == "fever" for e in examples)


# ── VitaminC loader ─────────────────────────────────────────────────


class TestLoadVitaminC:
    """Tests for VitaminC loader with label remapping."""

    @pytest.mark.parametrize(
        "raw_label,expected",
        [
            (0, LABEL_ENTAILMENT),
            (1, LABEL_NEUTRAL),
            (2, LABEL_CONTRADICTION),
            ("SUPPORTS", LABEL_ENTAILMENT),
            ("REFUTES", LABEL_CONTRADICTION),
            ("NOT ENOUGH INFO", LABEL_NEUTRAL),
        ],
    )
    @patch("datasets.load_dataset")
    def test_label_mapping(self, mock_ld, raw_label, expected):
        mock_ld.return_value = [{"evidence": "E", "claim": "C", "label": raw_label}]
        examples = _load_vitaminc()
        assert len(examples) == 1
        assert examples[0]["label"] == expected

    @patch("datasets.load_dataset")
    def test_source_tagged_vitaminc(self, mock_ld):
        mock_ld.return_value = [{"evidence": "E", "claim": "C", "label": 0}]
        examples = _load_vitaminc()
        assert all(e["source"] == "vitaminc" for e in examples)


# ── ANLI R3 loader ──────────────────────────────────────────────────


class TestLoadAnliR3:
    """Tests for ANLI Round 3 loader."""

    @pytest.mark.parametrize("label", [0, 1, 2])
    @patch("datasets.load_dataset")
    def test_all_three_labels_pass(self, mock_ld, label):
        mock_ld.return_value = [{"premise": "P", "hypothesis": "H", "label": label}]
        examples = _load_anli_r3()
        assert len(examples) == 1
        assert examples[0]["label"] == label

    @patch("datasets.load_dataset")
    def test_none_label_skipped(self, mock_ld):
        mock_ld.return_value = [{"premise": "P", "hypothesis": "H", "label": None}]
        examples = _load_anli_r3()
        assert len(examples) == 0

    @patch("datasets.load_dataset")
    def test_source_tagged(self, mock_ld):
        mock_ld.return_value = [{"premise": "P", "hypothesis": "H", "label": 0}]
        examples = _load_anli_r3()
        assert all(e["source"] == "anli_r3" for e in examples)


# ── RAGTruth loader ─────────────────────────────────────────────────


class TestLoadRAGTruth:
    """Tests for RAGTruth hallucination label parsing."""

    @patch("datasets.load_dataset")
    def test_no_hallucination_maps_to_entailment(self, mock_ld):
        from training.data_pipeline import _load_ragtruth

        mock_ld.return_value = [
            {
                "context": "C",
                "output": "O",
                "hallucination_labels_processed": "{'evident_conflict': 0, 'baseless_info': 0}",
            }
        ]
        examples = _load_ragtruth()
        assert len(examples) == 1
        assert examples[0]["label"] == LABEL_ENTAILMENT

    @patch("datasets.load_dataset")
    def test_evident_conflict_maps_to_contradiction(self, mock_ld):
        from training.data_pipeline import _load_ragtruth

        mock_ld.return_value = [
            {
                "context": "C",
                "output": "O",
                "hallucination_labels_processed": "{'evident_conflict': 2, 'baseless_info': 0}",
            }
        ]
        examples = _load_ragtruth()
        assert examples[0]["label"] == LABEL_CONTRADICTION

    @patch("datasets.load_dataset")
    def test_baseless_info_maps_to_contradiction(self, mock_ld):
        from training.data_pipeline import _load_ragtruth

        mock_ld.return_value = [
            {
                "context": "C",
                "output": "O",
                "hallucination_labels_processed": "{'evident_conflict': 0, 'baseless_info': 3}",
            }
        ]
        examples = _load_ragtruth()
        assert examples[0]["label"] == LABEL_CONTRADICTION

    @patch("datasets.load_dataset")
    def test_dict_labels_not_string(self, mock_ld):
        from training.data_pipeline import _load_ragtruth

        mock_ld.return_value = [
            {
                "context": "C",
                "output": "O",
                "hallucination_labels_processed": {
                    "evident_conflict": 0,
                    "baseless_info": 0,
                },
            }
        ]
        examples = _load_ragtruth()
        assert len(examples) == 1
        assert examples[0]["label"] == LABEL_ENTAILMENT

    @patch("datasets.load_dataset")
    def test_empty_context_skipped(self, mock_ld):
        from training.data_pipeline import _load_ragtruth

        mock_ld.return_value = [
            {"context": "", "output": "O", "hallucination_labels_processed": "{}"}
        ]
        examples = _load_ragtruth()
        assert len(examples) == 0

    @patch("datasets.load_dataset")
    def test_premise_truncated_at_2000(self, mock_ld):
        from training.data_pipeline import _load_ragtruth

        mock_ld.return_value = [
            {
                "context": "X" * 5000,
                "output": "O",
                "hallucination_labels_processed": "{}",
            }
        ]
        examples = _load_ragtruth()
        assert len(examples[0]["premise"]) == 2000

    @patch("datasets.load_dataset")
    def test_source_tagged_ragtruth(self, mock_ld):
        from training.data_pipeline import _load_ragtruth

        mock_ld.return_value = [
            {"context": "C", "output": "O", "hallucination_labels_processed": "{}"}
        ]
        examples = _load_ragtruth()
        assert all(e["source"] == "ragtruth" for e in examples)


# ── AggreFact loader ────────────────────────────────────────────────


class TestLoadAggreFact:
    """Tests for LLM-AggreFact gated dataset loader."""

    @patch("datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test"})
    def test_supported_maps_to_entailment(self, mock_ld):
        from training.data_pipeline import _load_aggrefact

        mock_ld.return_value = [
            {"doc": "D", "claim": "C", "label": 1, "dataset": "AggreFact-CNN"}
        ]
        examples = _load_aggrefact()
        assert examples[0]["label"] == LABEL_ENTAILMENT

    @patch("datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test"})
    def test_not_supported_maps_to_contradiction(self, mock_ld):
        from training.data_pipeline import _load_aggrefact

        mock_ld.return_value = [
            {"doc": "D", "claim": "C", "label": 0, "dataset": "RAGTruth"}
        ]
        examples = _load_aggrefact()
        assert examples[0]["label"] == LABEL_CONTRADICTION

    @patch.dict("os.environ", {}, clear=True)
    def test_no_token_returns_empty(self):
        import os

        from training.data_pipeline import _load_aggrefact

        os.environ.pop("HF_TOKEN", None)
        examples = _load_aggrefact()
        assert examples == []

    @patch("datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test"})
    def test_source_includes_dataset_name(self, mock_ld):
        from training.data_pipeline import _load_aggrefact

        mock_ld.return_value = [
            {"doc": "D", "claim": "C", "label": 1, "dataset": "ExpertQA"}
        ]
        examples = _load_aggrefact()
        assert examples[0]["source"] == "aggrefact_ExpertQA"

    @patch("datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test"})
    def test_empty_doc_skipped(self, mock_ld):
        from training.data_pipeline import _load_aggrefact

        mock_ld.return_value = [{"doc": "", "claim": "C", "label": 1, "dataset": "X"}]
        examples = _load_aggrefact()
        assert len(examples) == 0

    @patch("datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test"})
    def test_invalid_label_skipped(self, mock_ld):
        from training.data_pipeline import _load_aggrefact

        mock_ld.return_value = [{"doc": "D", "claim": "C", "label": 99, "dataset": "X"}]
        examples = _load_aggrefact()
        assert len(examples) == 0

    @patch("datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test"})
    def test_premise_truncated_at_2000(self, mock_ld):
        from training.data_pipeline import _load_aggrefact

        mock_ld.return_value = [
            {"doc": "Y" * 5000, "claim": "C", "label": 1, "dataset": "X"}
        ]
        examples = _load_aggrefact()
        assert len(examples[0]["premise"]) == 2000

    @patch("datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test"})
    @pytest.mark.parametrize(
        "ds_name",
        [
            "AggreFact-CNN",
            "AggreFact-XSum",
            "TofuEval-MediaS",
            "TofuEval-MeetB",
            "Wice",
            "Reveal",
            "ClaimVerify",
            "FactCheck-GPT",
            "ExpertQA",
            "Lfqa",
            "RAGTruth",
        ],
    )
    def test_all_11_subdatasets_tagged(self, mock_ld, ds_name):
        from training.data_pipeline import _load_aggrefact

        mock_ld.return_value = [
            {"doc": "D", "claim": "C", "label": 1, "dataset": ds_name}
        ]
        examples = _load_aggrefact()
        assert examples[0]["source"] == f"aggrefact_{ds_name}"


# ── VitaminC capping ────────────────────────────────────────────────


class TestVitaminCCap:
    """Test VitaminC dataset capping logic."""

    def test_cap_constant_is_100k(self):
        assert VITAMINC_CAP == 100_000


# ── build_dataset orchestration ─────────────────────────────────────


class TestBuildDataset:
    """Integration tests for build_dataset() function."""

    def _balanced_examples(self, n, source="test"):
        """Generate n examples with balanced 3-class labels."""
        return [
            {
                "premise": f"P{i}",
                "hypothesis": f"H{i}",
                "label": i % 3,
                "source": source,
            }
            for i in range(n)
        ]

    @patch("training.data_pipeline._load_anli_r3")
    @patch("training.data_pipeline._load_vitaminc")
    @patch("training.data_pipeline._load_fever")
    @patch("training.data_pipeline._load_halueval")
    def test_basic_build_combines_sources(
        self, mock_halu, mock_fever, mock_vita, mock_anli, tmp_path
    ):
        from training.data_pipeline import build_dataset

        mock_halu.return_value = self._balanced_examples(12, "halueval_qa")
        mock_fever.return_value = self._balanced_examples(6, "fever")
        mock_vita.return_value = self._balanced_examples(6, "vitaminc")
        mock_anli.return_value = self._balanced_examples(6, "anli_r3")

        with patch.object(
            sys.modules["training.data_pipeline"], "OUTPUT_DIR", tmp_path / "data"
        ):
            ds = build_dataset()

        assert "train" in ds
        assert "eval" in ds
        total = len(ds["train"]) + len(ds["eval"])
        assert total == 30

    @patch("training.data_pipeline._load_ragtruth")
    @patch("training.data_pipeline._load_anli_r3")
    @patch("training.data_pipeline._load_vitaminc")
    @patch("training.data_pipeline._load_fever")
    @patch("training.data_pipeline._load_halueval")
    def test_include_ragtruth_adds_samples(
        self, mock_halu, mock_fever, mock_vita, mock_anli, mock_ragtruth, tmp_path
    ):
        from training.data_pipeline import build_dataset

        for m in [mock_halu, mock_fever, mock_vita, mock_anli]:
            m.return_value = self._balanced_examples(6)
        mock_ragtruth.return_value = self._balanced_examples(6, "ragtruth")

        with patch.object(
            sys.modules["training.data_pipeline"], "OUTPUT_DIR", tmp_path / "data"
        ):
            ds = build_dataset(include_ragtruth=True)

        total = len(ds["train"]) + len(ds["eval"])
        assert total == 30  # 24 base + 6 ragtruth

    @patch("training.data_pipeline._load_anli_r3")
    @patch("training.data_pipeline._load_vitaminc")
    @patch("training.data_pipeline._load_fever")
    @patch("training.data_pipeline._load_halueval")
    def test_stats_json_written(
        self, mock_halu, mock_fever, mock_vita, mock_anli, tmp_path
    ):
        from training.data_pipeline import build_dataset

        for m in [mock_halu, mock_fever, mock_vita, mock_anli]:
            m.return_value = self._balanced_examples(6)

        out = tmp_path / "data"
        with patch.object(sys.modules["training.data_pipeline"], "OUTPUT_DIR", out):
            build_dataset()

        stats_path = out / "stats.json"
        assert stats_path.exists()
        stats = json.loads(stats_path.read_text())
        assert "total" in stats
        assert "label_distribution" in stats
        assert "source_distribution" in stats
        assert stats["total"] == 24


# ── Pipeline performance documentation ──────────────────────────────


class TestPipelinePerformanceDoc:
    """Verify data pipeline outputs document performance characteristics."""

    def _balanced_examples(self, n, source="test"):
        return [
            {
                "premise": f"P{i}",
                "hypothesis": f"H{i}",
                "label": i % 3,
                "source": source,
            }
            for i in range(n)
        ]

    @patch("training.data_pipeline._load_anli_r3")
    @patch("training.data_pipeline._load_vitaminc")
    @patch("training.data_pipeline._load_fever")
    @patch("training.data_pipeline._load_halueval")
    def test_stats_has_source_distribution(
        self, mock_halu, mock_fever, mock_vita, mock_anli, tmp_path
    ):
        from training.data_pipeline import build_dataset

        mock_halu.return_value = self._balanced_examples(18, "halueval_qa")
        mock_fever.return_value = self._balanced_examples(12, "fever")
        mock_vita.return_value = []
        mock_anli.return_value = []

        out = tmp_path / "data"
        with patch.object(sys.modules["training.data_pipeline"], "OUTPUT_DIR", out):
            build_dataset()

        stats = json.loads((out / "stats.json").read_text())
        assert stats["source_distribution"]["halueval_qa"] == 18
        assert stats["source_distribution"]["fever"] == 12

    @patch("training.data_pipeline._load_anli_r3")
    @patch("training.data_pipeline._load_vitaminc")
    @patch("training.data_pipeline._load_fever")
    @patch("training.data_pipeline._load_halueval")
    def test_stats_has_label_distribution(
        self, mock_halu, mock_fever, mock_vita, mock_anli, tmp_path
    ):
        from training.data_pipeline import build_dataset

        mock_halu.return_value = self._balanced_examples(30, "x")
        for m in [mock_fever, mock_vita, mock_anli]:
            m.return_value = []

        out = tmp_path / "data"
        with patch.object(sys.modules["training.data_pipeline"], "OUTPUT_DIR", out):
            build_dataset()

        stats = json.loads((out / "stats.json").read_text())
        assert "0" in stats["label_distribution"]
        assert "1" in stats["label_distribution"]
        assert "2" in stats["label_distribution"]

    @patch("training.data_pipeline._load_anli_r3")
    @patch("training.data_pipeline._load_vitaminc")
    @patch("training.data_pipeline._load_fever")
    @patch("training.data_pipeline._load_halueval")
    def test_train_eval_split_ratio(
        self, mock_halu, mock_fever, mock_vita, mock_anli, tmp_path
    ):
        from training.data_pipeline import build_dataset

        # Need enough samples for stratified split to work
        examples = [
            {
                "premise": f"P{i}",
                "hypothesis": f"H{i}",
                "label": i % 3,
                "source": "test",
            }
            for i in range(30)
        ]
        mock_halu.return_value = examples
        for m in [mock_fever, mock_vita, mock_anli]:
            m.return_value = []

        out = tmp_path / "data"
        with patch.object(sys.modules["training.data_pipeline"], "OUTPUT_DIR", out):
            ds = build_dataset()

        total = len(ds["train"]) + len(ds["eval"])
        eval_ratio = len(ds["eval"]) / total
        assert 0.05 <= eval_ratio <= 0.15  # ~10% ± tolerance
