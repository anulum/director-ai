# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AggreFact save-scores latency test
import json

# Add benchmarks to path
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("sklearn", reason="sklearn required for aggrefact_eval")

sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.aggrefact_eval import (  # noqa: E402
    _uses_factcg_template,
    score_and_save,
)


class TestAggrefactSaveScores(unittest.TestCase):
    @patch("benchmarks.aggrefact_eval._BinaryNLIPredictor")
    @patch("benchmarks.aggrefact_eval._load_aggrefact")
    def test_score_and_save_includes_latencies(self, mock_load, mock_predictor_cls):
        # Setup mock dataset
        mock_load.return_value = [
            {"doc": "Context 1", "claim": "Claim 1", "label": 1, "dataset": "ds1"},
            {"doc": "Context 2", "claim": "Claim 2", "label": 0, "dataset": "ds1"},
        ]

        # Setup mock predictor
        mock_predictor = MagicMock()
        mock_predictor.score.return_value = 0.8
        mock_predictor_cls.return_value = mock_predictor

        output_path = Path("test_scores.json")
        try:
            score_and_save(output_path, max_samples=2)

            with open(output_path) as f:
                data = json.load(f)

            self.assertIn("latencies_per_sample", data)
            self.assertEqual(len(data["latencies_per_sample"]), 2)
            self.assertIn("scores", data)
            self.assertIn("labels", data)
            self.assertIn("datasets_per_sample", data)

        finally:
            if output_path.exists():
                output_path.unlink()

    @patch("benchmarks.aggrefact_eval._BinaryNLIPredictor")
    @patch("benchmarks.aggrefact_eval._load_aggrefact")
    def test_score_and_save_passes_explicit_scorer_template(
        self,
        mock_load,
        mock_predictor_cls,
    ):
        mock_load.return_value = [
            {"doc": "Context 1", "claim": "Claim 1", "label": 1, "dataset": "ds1"},
            {"doc": "Context 2", "claim": "Claim 2", "label": 0, "dataset": "ds1"},
        ]
        mock_predictor = MagicMock()
        mock_predictor.score.side_effect = [0.9, 0.1]
        mock_predictor_cls.return_value = mock_predictor

        output_path = Path("test_scores_template.json")
        try:
            score_and_save(
                output_path,
                max_samples=2,
                model_name="/workspace/cache/resolved-model",
                scorer_template="factcg",
            )

            mock_predictor_cls.assert_called_once_with(
                model_name="/workspace/cache/resolved-model",
                max_length=2048,
                scorer_template="factcg",
            )
        finally:
            if output_path.exists():
                output_path.unlink()


def test_explicit_factcg_template_survives_resolved_cache_path():
    assert _uses_factcg_template(
        "/workspace/cache/huggingface/director-ai-scorers/resolved-model",
        object(),
        "factcg",
    )


def test_explicit_sequence_pair_template_overrides_factcg_name():
    assert not _uses_factcg_template(
        "yaxili96/FactCG-DeBERTa-v3-Large",
        object(),
        "sequence-pair",
    )


if __name__ == "__main__":
    unittest.main()
