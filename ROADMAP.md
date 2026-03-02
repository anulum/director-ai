# Roadmap

## v2.3.0 (current)

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

## v2.4.0 (planned)

- Rust-accelerated scorer backend via backfire-kernel FFI
- WebSocket multiplexed streaming (multiple sessions per connection)
- Model-agnostic ONNX quantization pipeline (INT8/FP16 auto-select)

## v3.0 (vision)

- **Simplified public API**: `guard()` as the primary interface; enterprise behind `director_ai.enterprise`
- **Rust-accelerated scorer**: PyO3 binding to backfire-ffi for sub-1ms heuristic scoring
- **Adaptive threshold calibration**: `director-ai tune` with labeled data → optimal threshold + weights
- **Provider-agnostic guard()**: duck-type detection supports any OpenAI-compatible client (done in 2.3.0)
- **WebSocket multiplexed streaming**: multiple concurrent sessions per connection with back-pressure
- **ONNX INT8/FP16 auto-select**: quantization pipeline picks precision based on hardware capability
- **Remove deprecated 1.x aliases**: `DirectorModule`, `BackfireKernel`, `StrangeLoopAgent`, etc.
- **Drop Python 3.10**: minimum Python 3.11 for `ExceptionGroup` and `TaskGroup` support
