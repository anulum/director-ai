"""Minimal Director-AI guard — run: python guard.py"""

from pathlib import Path

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.config import DirectorConfig

_HERE = Path(__file__).resolve().parent
config = DirectorConfig.from_yaml(str(_HERE / "config.yaml"))
store = GroundTruthStore()
with open(_HERE / "facts.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            store.add(line[:20], line)

scorer = CoherenceScorer(
    threshold=config.coherence_threshold,
    ground_truth_store=store,
    use_nli=config.use_nli,
)

approved, score = scorer.review("What color is the sky?", "The sky is blue.")
print(f"Approved: {approved}  Score: {score.score:.3f}")
