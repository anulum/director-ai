# Director-AI — Launch Post Drafts

## Show HN

**Title:** Director-AI – Real-time LLM hallucination guardrail with streaming halt

**Body:**

I built an open-source guardrail that sits between your LLM and the user.
It scores every output for hallucination (NLI contradiction detection +
RAG fact-checking against your own KB) and can halt generation mid-stream
if coherence drops below threshold.

Three things make it different from existing tools:

1. Token-level streaming halt. Not post-hoc review — the safety kernel
   monitors coherence token-by-token and severs output the moment it
   degrades. Three halt mechanisms: hard limit, sliding window, downward
   trend.

2. Dual-entropy scoring. NLI (DeBERTa) catches logical contradictions.
   RAG retrieval catches factual deviation from your knowledge base.
   Both must pass.

3. Your data, your rules. Ingest your own documents into ChromaDB.
   The scorer checks against your ground truth, not a generic model.

It's ~1 MB base install (numpy + requests). NLI model is optional
(`pip install director-ai[nli]`). Works with any OpenAI-compatible
endpoint (llama.cpp, vLLM, Ollama, etc.).

75.8% balanced accuracy on AggreFact (29K samples) — 4th on the leaderboard,
within 1.6pp of the top 7B model at 17x fewer params. And the only tool
with real-time streaming halt. NLI component is pluggable.

Honest benchmarks comparing to MiniCheck, HHEM, and others in the repo.

- GitHub: https://github.com/anulum/director-ai
- PyPI: `pip install director-ai`
- Demo: https://huggingface.co/spaces/anulum/director-ai
- License: AGPL-3.0 (commercial licensing available)

---

## r/LocalLLaMA

**Title:** Director-AI: open-source guardrail that halts hallucinating LLMs mid-stream (works with llama.cpp, Ollama, vLLM)

**Body:**

Released Director-AI — a hallucination guardrail that sits between your
local LLM and the user.

What it does:
- Scores every response for coherence (NLI + RAG against your KB)
- Halts generation token-by-token if coherence drops (not post-hoc)
- Works with any OpenAI-compatible endpoint

Quick example:
```python
pip install director-ai
```
```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("refund policy", "Refunds within 30 days only.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
approved, score = scorer.review(
    "What is the refund policy?",
    "We offer full refunds within 90 days."  # wrong
)
# approved = False
```

Base install is ~1 MB (no torch). Add NLI with `pip install director-ai[nli]`.

Benchmarked on AggreFact: 75.8% balanced accuracy (4th on leaderboard).
And no other tool does real-time streaming halt. The NLI model is
pluggable, so swap in whatever scores best for your domain.

GitHub: https://github.com/anulum/director-ai

---

## r/MachineLearning

**Title:** [P] Director-AI: Token-level streaming halt for LLM hallucination (NLI + RAG, DeBERTa, open-source)

**Body:**

Director-AI is a hallucination guardrail with token-level streaming
oversight. It combines NLI contradiction detection (DeBERTa-v3) with
RAG fact-checking against a user-provided knowledge base, and can halt
generation mid-stream via three mechanisms: hard coherence floor,
sliding window average, and downward trend detection.

Scoring formula: `Coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)`

Evaluated on LLM-AggreFact (29,320 samples, 11 datasets):
- FactCG-DeBERTa-v3-Large (default): 75.8% balanced accuracy
- Best per-dataset: 87.3% on Lfqa, 86.2% on TofuEval-MediaS

Ranks 4th on the leaderboard behind Bespoke-MiniCheck-7B (77.4%) at
17x fewer params. And the only tool with real-time streaming halt.
NLI component is pluggable.

Fine-tuning pipeline (DeBERTa on AggreFact) included in `training/`.
The NLI scorer accepts custom model paths.

- GitHub: https://github.com/anulum/director-ai
- PyPI: `pip install director-ai`
- Benchmarks: `benchmarks/` directory with full comparison
- License: AGPL-3.0

---

## Twitter/X

Director-AI — real-time LLM hallucination guardrail.

NLI + RAG scoring. Token-level streaming halt (not post-hoc).
Your knowledge base, your rules.

~1 MB install. Works with llama.cpp, Ollama, vLLM, OpenAI.

pip install director-ai
https://github.com/anulum/director-ai
