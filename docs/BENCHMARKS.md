# Director-AI — Performance Benchmarks

## Scoring Tiers

Director-AI uses a 5-tier scoring pyramid. Each tier trades latency for accuracy.

| Tier | Backend | BA (AggreFact) | Latency | Params | Notes |
|------|---------|---------------|---------|--------|-------|
| 1 | Heuristic keywords | ~55% | <1ms | 0 | Regex patterns, no model |
| 2 | Rules engine | ~58% | <1ms | 0 | Configurable rule chains |
| 3 | Embedding similarity | ~65% | 5ms | varies | SentenceTransformers cosine |
| 4 | Distilled NLI (DeBERTa-v3-xsmall) | *pending* | 27ms CPU | 70M | Knowledge-distilled from Tier 5 |
| 5 | Full NLI (FactCG-DeBERTa-v3-Large) | 75.8% | 574ms GPU | 400M | Reference model |

### Full NLI (Tier 5) — AggreFact Breakdown

Model: `yaxili96/FactCG-DeBERTa-v3-Large` at threshold 0.46.

| Dataset | Samples | BA |
|---------|---------|-----|
| AggreFact-CNN | 558 | 68.8% |
| AggreFact-XSum | 558 | 74.3% |
| ClaimVerify | 1,088 | 78.1% |
| ExpertQA | 3,702 | 59.1% |
| FactCheck-GPT | 1,566 | 73.0% |
| Lfqa | 1,911 | 86.4% |
| RAGTruth | 16,371 | 82.2% |
| Reveal | 1,710 | 89.1% |
| TofuEval-MediaS | 726 | 71.9% |
| TofuEval-MeetB | 772 | 74.3% |
| Wice | 358 | 76.9% |
| **Macro Average** | **29,320** | **75.8%** |

### Distilled NLI (Tier 4)

Model: `anulum/director-ai-nli-lite` (DeBERTa-v3-xsmall, 70M params).
Training: 10 epochs, KL divergence + hard label blend (alpha=0.3), T=1.5.

- ONNX INT8 latency: 27ms CPU
- Model size: 83MB ONNX
- Sanity checks (6/6 PASS):
  - Supported claims: P[support] = 0.84–0.87
  - Contradicted claims: P[contra] = 0.74–0.89
- Full AggreFact evaluation: *in progress*

---

## RAG Backends

All RAG techniques are independently toggleable decorator backends.
Measured on a synthetic 10K-document knowledge base.

| Backend | Operation | Latency Overhead | Memory Overhead |
|---------|-----------|-----------------|-----------------|
| BM25 Hybrid (RRF) | Query-time | +2–5ms | +index size |
| Cross-encoder reranker | Query-time | +15–50ms | +model load |
| Parent-child chunking | Index + query | +1ms query | +parent storage |
| Adaptive retrieval routing | Query-time | <1ms (heuristic) | negligible |
| HyDE (pseudo-doc) | Query-time | +LLM call | negligible |
| Query decomposition | Query-time | +LLM call per sub-query | negligible |
| Contextual compression | Query-time | +1–2ms (keyword), +LLM (advanced) | negligible |
| Multi-vector | Index + query | +2–3ms query | +summary/title vectors |

### RAG Decorator Stack

```
base (Chroma/Pinecone/FAISS/...)
  → HybridBackend (BM25+dense, RRF)
    → RerankedBackend (cross-encoder)
      → ParentChildBackend (return parent from child match)
        → HyDEBackend (pseudo-doc generation before embed)
          → QueryDecompositionBackend (split → parallel retrieve → merge)
            → ContextualCompressionBackend (LLM compress results)
```

All layers are enabled via config fields (e.g., `parent_child_enabled: true`).

---

## Agentic Swarm

Performance characteristics for SwarmGuardian multi-agent monitoring.

| Operation | Latency | Thread-safe |
|-----------|---------|-------------|
| Agent registration | <0.1ms | Yes (Lock) |
| Handoff scoring (keyword) | <0.5ms | Yes |
| Handoff scoring (NLI) | +NLI latency | Yes |
| Cascade halt propagation | O(n) agents | Yes |
| Metrics query | <0.1ms | Yes (Lock) |

### Framework Adapters

| Framework | Integration Point | Overhead |
|-----------|------------------|----------|
| LangGraph | Conditional edge | <0.5ms per edge |
| CrewAI | Task output callback | <0.5ms per task |
| OpenAI Swarm | Function wrapper | <0.5ms per handoff |
| AutoGen | Group chat filter | <0.5ms per message |

---

## Configuration

| Feature | Config field | Default |
|---------|-------------|---------|
| Scorer backend | `scorer_backend` | `auto` (rust > onnx > deberta > lite) |
| Hardened mode | `hardened` | `false` |
| Dry-run mode | `dry_run` | `false` |
| Production mode | `production_mode` | `false` |
| Cost tracking | `cost_tracking_enabled` | `false` |

---

## Test Coverage

| Category | Test files | Tests |
|----------|-----------|-------|
| Core scoring | 8 | ~120 |
| RAG backends | 6 | ~140 |
| Agentic/swarm | 5 | ~100 |
| CLI commands | 3 | ~80 |
| Compliance | 2 | ~40 |
| Config | 2 | ~30 |
| **Total** | **26+** | **500+** |

---

*Generated from verified benchmark runs. Numbers are from local evaluation
on the mining rig (i5-11600K, GTX 1060 6GB / 5x RX 6600 XT 8GB).
AggreFact results are on the full 29,320-sample LLM-AggreFact benchmark.*
