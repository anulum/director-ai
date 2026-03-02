# Roadmap

## v2.2.0 (current)

### Done
- `score_every_n`, `adaptive`, `max_cadence` on StreamingKernel + AsyncStreamingKernel
- Runtime validation on threshold, soft_limit, w_logic, w_fact
- Streaming overhead benchmark (tokens/sec by cadence)
- Enterprise modules lazy-loaded via `__getattr__`
- `[enterprise]` optional dependency group + pytest marker
- Stdlib imports hoisted in scorer.py and streaming.py

## v2.1.0

### Done
- `director-ai bench` CLI subcommand (--dataset, --seed, --output)
- `scorer_backend="hybrid"` mode (NLI + LLM judge)
- Architecture deep-dive doc (guide/architecture.md)
- Production checklist doc (deployment/checklist.md)
- Threshold tuning guide expanded
- 35 streaming false-halt benchmark passages
- PineconeBackend, WeaviateBackend, QdrantBackend
- Bandit + Semgrep SAST in CI

## v2.0.0

### Done
- Case-sensitivity fix in GroundTruthStore
- LLM judge error handling hardened
- SafetyKernel hard_limit validation
- Thread-safe OTel setup
- Histogram bucket_counts O(n log n) optimization
- 12 fragile inspect.getsource tests replaced

## v2.3.0 (planned)

- Multi-turn conversation tracking (session state across reviews)
- Lite scorer (no-NLI fast path with learned heuristics)
- ONNX GPU batch inference optimization
- Stable public API freeze

## v2.4.0 (planned)

- Plugin architecture for custom scorer backends
- gRPC transport for server.py
- Multi-GPU sharding for NLI inference
- Third-party security audit
