# Roadmap

## v2.6.0 (current)

### Done
- StreamingKernel wired into `CoherenceAgent.stream()` for unified token-level oversight
- gRPC incremental streaming with per-chunk coherence scores
- CLI multi-worker config propagation (`--workers`)
- ONNX batch config wiring end-to-end (`onnx_path` in DirectorConfig)
- Prompt content removed from logs — HMAC audit hashing
- `guard()` duck-type detection for OpenAI/Anthropic SDK interceptor
- `strict_mode` reject in `CoherenceScorer.review()`
- Domain profiles: medical, finance, legal, creative, customer_support, summarization
- Discord bot + CI/release webhook automation
- E2E benchmark context leakage fix (per-sample store isolation)
- InputSanitizer additive scoring model fix

## v2.5.0

### Done
- Domain-specific scoring profiles (medical, finance, legal, creative, customer_support, summarization)
- `strict_mode` reject in CoherenceScorer
- `guard()` SDK interceptor with duck-type provider detection
- Persistent stats backend (SQLite)
- AGPL §13 `/v1/source` compliance endpoint

## v2.3.0

### Done
- Lite scorer backend (`scorer_backend="lite"`) — word overlap + negation heuristics, ~0.5ms/pair
- Multi-turn conversation tracking (`ConversationSession`) with cross-turn divergence blending
- ONNX GPU batch optimization (`OnnxDynamicBatcher`) with IO binding for zero-copy transfers
- Plugin architecture for scorer backends (`ScorerBackend` ABC + entry-point registry)
- gRPC transport (`proto/director.proto`, `--transport grpc` on CLI)
- Multi-GPU sharding (`ShardedNLIScorer`) with round-robin device routing
- Security audit preparation: threat model, SBOM generation, Hypothesis fuzz tests, `InputSanitizer` hardening
- Public API freeze: `__all__` on all modules, deprecated aliases emit `DeprecationWarning`

## v2.2.1

### Done
- API autodoc pages for DirectorConfig, Enterprise, InputSanitizer
- Troubleshooting guide, enterprise guide, streaming cadence examples
- Validation rules section in scorer reference

## v2.2.0

### Done
- `score_every_n`, `adaptive`, `max_cadence` on StreamingKernel + AsyncStreamingKernel
- Runtime validation on threshold, soft_limit, w_logic, w_fact
- Streaming overhead benchmark (tokens/sec by cadence)
- Enterprise modules lazy-loaded via `__getattr__`
- `[enterprise]` optional dependency group + pytest marker

## v2.1.0

### Done
- `director-ai bench` CLI subcommand (--dataset, --seed, --output)
- `scorer_backend="hybrid"` mode (NLI + LLM judge)
- Architecture deep-dive, production checklist, threshold tuning docs
- PineconeBackend, WeaviateBackend, QdrantBackend
- Bandit + Semgrep SAST in CI

## v2.0.0

### Done
- Case-sensitivity fix in GroundTruthStore
- LLM judge error handling hardened
- SafetyKernel hard_limit validation
- Thread-safe OTel setup
- Histogram bucket_counts O(n log n) optimization

## v2.8.0

### Done
- Rust-accelerated scorer backend wired into `CoherenceScorer(scorer_backend="rust")`
- WebSocket multiplexed streaming (concurrent sessions per connection, cancel, backpressure)
- VectorBackend entry-point registry (`register_vector_backend`, `get_vector_backend`, `list_vector_backends`)
- Tenant-isolated VectorStores via `TenantRouter.get_vector_store()`
- `/v1/tenants/{tenant_id}/vector-facts` REST endpoint
- ONNX INT8/FP16 quantization shipped in v2.3.0 (retired from planned)

## v3.0 (vision)

- **Simplified public API**: `guard()` as the primary interface; enterprise behind `director_ai.enterprise`
- **Adaptive threshold calibration**: `director-ai tune` with labeled data → optimal threshold + weights
- **Remove deprecated 1.x aliases**: `DirectorModule`, `BackfireKernel`, `StrangeLoopAgent`, etc.
- **Drop Python 3.10**: minimum Python 3.11 for `ExceptionGroup` and `TaskGroup` support
