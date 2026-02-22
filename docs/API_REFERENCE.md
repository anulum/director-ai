# Director-Class AI — API Reference

**Version**: 0.7.x
**Updated**: 2026-02-22

---

## Core Package (`director_ai.core`)

### CoherenceAgent

Main orchestrator pipeline. Generates candidate responses, scores them, and
gates output through the Safety Kernel.

```python
from director_ai.core import CoherenceAgent

agent = CoherenceAgent()
result = agent.process("What is the color of the sky?")
print(result.output, result.coherence.score)
```

#### Methods

- `process(prompt) -> ReviewResult` — Run full pipeline (generate, score, gate)
- `process_query(prompt)` — *Deprecated alias for `process()`*

### CoherenceScorer

Dual-entropy scorer combining logical (NLI) and factual (RAG) divergence.

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
approved, score = scorer.review("prompt", "response")
```

#### Methods

- `review(prompt, response) -> tuple[bool, CoherenceScore]` — Score and approve/reject
- `calculate_logical_divergence(prompt, response) -> float` — NLI contradiction score
- `calculate_factual_divergence(prompt, response) -> float` — RAG deviation score

### SafetyKernel

Software safety gate — halts token streams when coherence drops below threshold.

```python
from director_ai.core import SafetyKernel

kernel = SafetyKernel(emergency_threshold=0.5)
for token in kernel.stream_output(token_gen, sec_callback):
    print(token, end="")
```

#### Methods

- `stream_output(token_generator, sec_callback) -> Generator[str]` — Stream with safety gate
- `emergency_stop()` — Immediately sever the output stream

### StreamingKernel

Token-by-token oversight with three halt mechanisms: hard limit, sliding window
average, and downward trend detection.

```python
from director_ai.core import StreamingKernel

kernel = StreamingKernel(threshold=0.5)
for event in kernel.stream_tokens(tokens, scorer):
    print(event.token, event.score)
```

### MockGenerator / LLMGenerator

Candidate response generators.

```python
from director_ai.core import LLMGenerator

gen = LLMGenerator(api_url="http://localhost:8080/completion")
candidates = gen.generate_candidates(prompt, n=3)
```

#### Methods

- `generate_candidates(prompt, n=3) -> list[str]` — Generate n candidate responses

### GroundTruthStore

RAG ground-truth retrieval for factual divergence scoring.

```python
from director_ai.core import GroundTruthStore

store = GroundTruthStore()
context = store.retrieve_context("How many layers does SCPN have?")
```

### BatchProcessor

Bulk processing with configurable concurrency and timeout.

```python
from director_ai.core import CoherenceAgent, BatchProcessor

agent = CoherenceAgent()
processor = BatchProcessor(agent, item_timeout=30.0, max_concurrency=4)
result = processor.process_batch(["prompt1", "prompt2", ...])
print(result.total, result.succeeded, result.failed)
```

### DirectorConfig

Configuration management with profiles, environment variables, and YAML support.

```python
from director_ai.core import DirectorConfig

cfg = DirectorConfig.from_profile("thorough")
cfg = DirectorConfig.from_env()
```

### NLIScorer

NLI-based logical divergence scorer using DeBERTa-v3-base-mnli (lazy-loaded).

```python
from director_ai.core import NLIScorer

scorer = NLIScorer(use_model=True)
h_logical = scorer.score("The sky is blue", "The sky is green")  # ~0.9
```

### Data Types

- `CoherenceScore` — `score`, `h_logical`, `h_factual` fields
- `ReviewResult` — `output`, `halted`, `candidates_evaluated`, `coherence`

---

## Research Package (`director_ai.research`)

*Requires: `pip install director-ai[research]`*

### Physics (`research.physics`)

```python
from director_ai.research.physics import (
    SECFunctional,    # Lyapunov stability functional
    UPDEStepper,      # Euler-Maruyama UPDE integrator
    L16Controller,    # PI controllers with anti-windup
    build_knm_matrix, # Canonical 16x16 coupling matrix
    load_omega_n,     # 16 natural frequencies
)
```

### Consciousness (`research.consciousness`)

```python
from director_ai.research.consciousness import (
    TCBOObserver,     # Persistent homology -> p_h1
    TCBOController,   # PI feedback for gap-junction kappa
    PGBOEngine,       # Phase-to-Geometry Bridge Operator
)
```

### Consilium (`research.consilium`)

```python
from director_ai.research.consilium import ConsiliumAgent
agent = ConsiliumAgent()
decision = agent.decide()
```

---

## Backfire Kernel (Rust + PyO3)

The `backfire-kernel` workspace provides native Rust implementations with Python
bindings via PyO3. Install via: `pip install backfire-kernel`

```python
import backfire_kernel as bk

# Safety kernel
kernel = bk.SafetyKernel(emergency_threshold=0.5)

# Streaming kernel
stream = bk.StreamingKernel(threshold=0.5, window_size=10)

# SSGF engine
from backfire_kernel import SSGFConfig, GradientMethod
config = SSGFConfig(gradient_method=GradientMethod.Analytic)  # 7,609x faster
```

All operations complete within the **50 ms deadline** (Backfire Prevention Protocols §2.2).

---

## CLI

```bash
director-ai version                          # Show version
director-ai review "prompt" "response"       # Score a prompt/response pair
director-ai process "prompt"                 # Full pipeline processing
director-ai batch input.jsonl --output out   # Batch (max 10K, <100MB)
director-ai serve --port 8080 --profile fast # Start FastAPI server
director-ai config --profile thorough        # Show/set configuration
```

---

## REST API (FastAPI Server)

Start with `director-ai serve` or programmatically:

```python
from director_ai.server import create_app
app = create_app()
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/review` | Score a prompt/response pair |
| POST | `/v1/process` | Full pipeline processing |
| GET | `/v1/config` | Current configuration |
| GET | `/v1/health` | Health check |
| GET | `/v1/metrics` | Prometheus metrics |
| WS | `/v1/stream` | WebSocket token streaming |

---

## Legal & Attribution

**Organization**: Anulum CH&LI / Anulum Institute
**Author**: Miroslav Sotek — ORCID [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
**Copyright**: (C) 1998-2026 Miroslav Sotek. All rights reserved.
**Website**: [www.anulum.li](https://www.anulum.li)
**License**: GNU AGPL v3 | Commercial licensing available
