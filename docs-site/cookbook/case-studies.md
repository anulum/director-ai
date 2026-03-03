# Case Studies

Real-world deployment patterns with Director-AI.

## Legal RAG Assistant

**Setup**: 400-document contract knowledge base, ChromaDB backend, ONNX GPU.

```python
from director_ai import CoherenceScorer, VectorGroundTruthStore
from director_ai.core.vector_store import ChromaBackend

backend = ChromaBackend(
    collection_name="legal_contracts",
    persist_directory="/data/chroma",
    embedding_model="BAAI/bge-large-en-v1.5",
)
store = VectorGroundTruthStore(backend=backend)
store.ingest(["Contract clause 1...", "Contract clause 2..."])

scorer = CoherenceScorer(
    ground_truth_store=store,
    use_nli=True,
    threshold=0.7,
    nli_device="cuda",
)

approved, score = scorer.review("What is the liability cap?", llm_response)
```

**Results** (14-day deployment, 1,247 queries/day):

| Metric | Before Director-AI | After |
|--------|-------------------|-------|
| Hallucinated citations | 19% | 0.7% |
| False-halt rate | — | 0.9% |
| Median latency overhead | — | +11 ms |
| User satisfaction | 3.2/5 | 4.6/5 |

Key insight: setting `threshold=0.7` (above the default 0.6) eliminated
nearly all false citations. The 0.9% false-halt rate was on edge cases
where the model couldn't find supporting evidence in the KB — users
rated these as "barely noticeable".

## Finance Research Agent (CrewAI)

**Setup**: 8-step research pipeline via CrewAI.

```python
from crewai import Agent, Task, Crew
from director_ai.integrations.crewai import DirectorAITool

guardrail = DirectorAITool(
    facts={"SEC filing date": "2025-12-15", "quarterly revenue": "$4.2B"},
    threshold=0.65,
)

researcher = Agent(
    role="Financial Researcher",
    tools=[guardrail],
    goal="Verify all claims against SEC filings",
)
```

**Pattern**: Streaming halt on outdated SEC filing data triggers
automatic re-retrieval of the current filing.

## Creative Writing Co-Pilot

**Setup**: Long-form fiction with user-provided world bible as KB.

```python
from director_ai import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()  # empty — populate with your KB
store.add("protagonist", "Kael is a frost mage from the Northern Reach.")
store.add("magic system", "Only three schools of magic exist: frost, flame, void.")

scorer = CoherenceScorer(
    ground_truth_store=store,
    threshold=0.4,
    soft_limit=0.5,
    use_nli=True,
)

approved, score = scorer.review(
    "Describe Kael's abilities",
    draft_paragraph,
)
if score.warning:
    logger.warning("Low coherence: %s", score.score)
```

**Results**:

| Metric | Director-AI | Llama Guard 3 |
|--------|------------|---------------|
| False-halt rate (creative text) | 2.1% | 14% |
| Factual consistency with world bible | 94% | N/A (no KB) |
| Latency overhead | +15 ms | +300 ms |

Key insight: creative writing needs low thresholds (0.4) and
`soft_limit` warn-only mode for first drafts. Switch to `threshold=0.6`
for final consistency checks against the world bible.

## Deployment patterns

### Warn-only mode (development)

```python
from director_ai import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()  # empty — populate with your KB
scorer = CoherenceScorer(threshold=0.3, soft_limit=0.5, ground_truth_store=store)
approved, score = scorer.review(query, response)
if score.warning:
    logger.warning("Low coherence: %s", score.score)
```

### Strict mode (production)

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(
    threshold=0.6,
    strict_mode=True,
    use_nli=True,
)
```

### Agent with fallback (user-facing)

```python
from director_ai import CoherenceAgent

agent = CoherenceAgent(fallback="retrieval")
result = agent.process("What is the refund policy?")
if result.halted:
    print("Fell back to KB retrieval")
```
