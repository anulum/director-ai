# Case Studies

Real-world deployment patterns with Director-AI.

## Legal RAG Assistant

**Setup**: 400-document contract knowledge base, ChromaDB backend, ONNX GPU.

```python
from director_ai import CoherenceAgent, VectorGroundTruthStore, ChromaBackend

backend = ChromaBackend(
    collection_name="legal_contracts",
    persist_directory="/data/chroma",
    embedding_model="BAAI/bge-large-en-v1.5",
)
store = VectorGroundTruthStore(backend=backend)
store.ingest_directory("/data/contracts/", glob="*.pdf")

agent = CoherenceAgent(
    ground_truth_store=store,
    use_nli=True,
    threshold=0.7,       # strict for legal
    nli_device="cuda",
)
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

**Setup**: 8-step research → trade recommendation pipeline via CrewAI.

```python
from crewai import Agent, Task, Crew
from director_ai.integrations.crewai import DirectorAITool

guardrail = DirectorAITool(
    facts={"SEC filing date": "2025-12-15", "quarterly revenue": "$4.2B"},
    threshold=0.65,
    on_halt="fallback",  # re-retrieve instead of blocking
)

researcher = Agent(
    role="Financial Researcher",
    tools=[guardrail],
    goal="Verify all claims against SEC filings",
)
```

**Pattern**: Streaming halt on outdated SEC filing data triggers
automatic re-retrieval of the current filing. The `on_halt="fallback"`
mode re-queries the knowledge base instead of blocking the pipeline.

## Creative Writing Co-Pilot

**Setup**: Long-form fiction with user-provided world bible as KB.

```python
agent = CoherenceAgent(
    ground_truth_store=world_bible_store,
    threshold=0.4,       # permissive for creative text
    soft_limit=0.5,      # warn but don't halt
    use_nli=True,
)
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
agent = CoherenceAgent(threshold=0.3, soft_limit=0.5)
result = agent.process(query)
if result.coherence.warnings:
    logger.warning("Low coherence: %s", result.coherence.score)
# Never halts — just logs warnings
```

### Strict mode (production)

```python
agent = CoherenceAgent(
    threshold=0.6,
    strict_mode=True,  # no heuristic fallback
    use_nli=True,
)
```

### Fallback mode (user-facing)

```python
from director_ai import CoherenceAgent

agent = CoherenceAgent(
    threshold=0.5,
    fallback="retrieval",  # on halt: return top KB chunks instead
)
```
