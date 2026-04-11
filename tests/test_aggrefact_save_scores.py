import unittest
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add benchmarks to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "benchmarks"))

from aggrefact_eval import score_and_save

class TestAggrefactSaveScores(unittest.TestCase):
    @patch('aggrefact_eval._BinaryNLIPredictor')
    @patch('aggrefact_eval._load_aggrefact')
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
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn("latencies_per_sample", data)
            self.assertEqual(len(data["latencies_per_sample"]), 2)
            self.assertIn("scores", data)
            self.assertIn("labels", data)
            self.assertIn("datasets_per_sample", data)
            
        finally:
            if output_path.exists():
                output_path.unlink()

if __name__ == "__main__":
    unittest.main()
