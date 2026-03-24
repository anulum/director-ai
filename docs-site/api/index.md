# API Reference

Complete reference for every public class, function, and dataclass in Director-AI.

## Quick Navigation

### Entry Points

| Symbol | Module | Purpose |
|--------|--------|---------|
| [`guard()`](guard.md) | `director_ai` | Wrap an LLM SDK client with coherence scoring |
| [`score()`](guard.md#score) | `director_ai` | Score a single prompt/response pair |
| [`get_score()`](guard.md#get_score) | `director_ai` | Retrieve last score from `on_fail="metadata"` |
| [`VoiceGuard`](../guide/voice-ai.md) | `director_ai.integrations.voice` | Real-time token filter for voice AI / TTS pipelines |

### Core Classes

| Class | Module | Purpose |
|-------|--------|---------|
| [`CoherenceScorer`](scorer.md) | `director_ai.core.scoring.scorer` | Dual-entropy coherence scoring engine |
| [`StreamingKernel`](streaming.md) | `director_ai.core.runtime.streaming` | Token-level streaming halt |
| [`AsyncStreamingKernel`](streaming.md#async) | `director_ai.core.runtime.async_streaming` | Async variant of StreamingKernel |
| [`CoherenceAgent`](agent.md) | `director_ai.core.agent` | Orchestrator: generator + scorer + kernel |
| [`BatchProcessor`](batch.md) | `director_ai.core.runtime.batch` | Concurrent batch scoring |

### Knowledge & Retrieval

| Class | Module | Purpose |
|-------|--------|---------|
| [`GroundTruthStore`](guard.md) | `director_ai.core.retrieval.knowledge` | Key-value fact store (prototype) |
| [`VectorGroundTruthStore`](vector-store.md) | `director_ai.core.retrieval.vector_store` | Semantic vector store with pluggable backends |
| [`VectorBackend`](vector-store.md#vectorbackend) | `director_ai.core.retrieval.vector_store` | Abstract backend protocol |

### Configuration

| Class | Module | Purpose |
|-------|--------|---------|
| [`DirectorConfig`](config.md) | `director_ai.core.config` | Env var / YAML / profile configuration |

### Data Types

| Class | Module | Purpose |
|-------|--------|---------|
| [`CoherenceScore`](types.md) | `director_ai.core.types` | Score result with H_logical, H_factual, evidence |
| [`ReviewResult`](types.md#reviewresult) | `director_ai.core.types` | Agent review output |
| [`ScoringEvidence`](types.md#scoringevidence) | `director_ai.core.types` | Retrieved chunks + NLI details |
| [`HaltEvidence`](types.md#haltevidence) | `director_ai.core.types` | Structured halt reason with evidence |
| [`TokenEvent`](streaming.md#tokenevent) | `director_ai.core.runtime.streaming` | Per-token stream event |
| [`StreamSession`](streaming.md#streamsession) | `director_ai.core.runtime.streaming` | Complete stream session state |

### Interfaces

| Interface | Purpose |
|-----------|---------|
| [REST Server](server.md) | FastAPI endpoints (`/v1/review`, `/v1/health`, `/v1/metrics`) |
| [gRPC Server](grpc.md) | Protocol Buffers service (4 RPC methods) |
| [CLI](cli.md) | 15 command-line subcommands |

### Exceptions

| Exception | Raised When |
|-----------|-------------|
| [`HallucinationError`](exceptions.md) | `guard()` with `on_fail="raise"` detects low coherence |
| [`KernelHaltError`](exceptions.md#kernelhalterror) | SafetyKernel halts the output stream |
| [`ValidationError`](exceptions.md#validationerror) | Invalid configuration or input |
| [`DependencyError`](exceptions.md#dependencyerror) | Required optional package missing |

### Meta-Confidence & Contradiction (v3.10.0)

| Symbol | Module | Purpose |
|--------|--------|---------|
| `compute_meta_confidence()` | `core.scoring.meta_confidence` | Verdict confidence from margin + signal agreement |
| `ContradictionTracker` | `core.runtime.contradiction_tracker` | Pairwise cross-turn contradiction matrix |
| `ContradictionReport` | `core.runtime.contradiction_tracker` | Contradiction summary (worst pair, trend) |

See [Meta-Confidence Guide](../guide/meta-confidence.md).

### Structured Output Verification (v3.10.0)

| Function | Module | Purpose |
|----------|--------|---------|
| `verify_json()` | `core.verification.json_verifier` | JSON Schema validation + value grounding |
| `verify_tool_call()` | `core.verification.tool_call_verifier` | Tool existence, arg validation, fabrication detection |
| `verify_code()` | `core.verification.code_verifier` | Python syntax, import existence, hallucinated API detection |

Result types: `StructuredVerificationResult`, `ToolCallResult`, `CodeCheckResult`, `FieldVerdict`.

See [Structured Verification Guide](../guide/structured-verification.md).

### Online Calibration (v3.10.0)

| Class | Module | Purpose |
|-------|--------|---------|
| `FeedbackStore` | `core.calibration.feedback_store` | SQLite-backed human correction store |
| `OnlineCalibrator` | `core.calibration.online_calibrator` | Threshold sweep + FPR/FNR with Wilson CIs |
| `CalibrationReport` | `core.calibration.online_calibrator` | Calibration metrics dataclass |

See [Online Calibration Guide](../guide/online-calibration.md).

## Import Patterns

```python
# Top-level convenience imports
from director_ai import guard, score, get_score
from director_ai import CoherenceScorer, StreamingKernel, CoherenceAgent

# v3.10.0: Structured verification (stdlib only, no torch)
from director_ai import verify_json, verify_tool_call, verify_code

# v3.10.0: Online calibration
from director_ai import FeedbackStore, OnlineCalibrator, CalibrationReport

# v3.10.0: Meta-confidence
from director_ai import compute_meta_confidence, ContradictionTracker

# Direct module imports (for type hints and advanced use)
from director_ai.core.config import DirectorConfig
from director_ai.core.types import CoherenceScore, ReviewResult
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore, ChromaBackend
from director_ai.core.runtime.batch import BatchProcessor

# Enterprise (lazy-loaded)
from director_ai.enterprise import TenantRouter, Policy, AuditLogger
```
