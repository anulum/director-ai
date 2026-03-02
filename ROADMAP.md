# Roadmap

## v2.1.0 (current)

### Done
- `director-ai bench` CLI subcommand (--dataset, --seed, --output)
- `scorer_backend="hybrid"` mode (NLI + LLM judge on every review)
- `scorer_backend` field on DirectorConfig
- Architecture deep-dive doc (guide/architecture.md)
- Production checklist doc (deployment/checklist.md)
- Threshold tuning guide expanded (grid-search, domain table, pitfalls)
- 35 new streaming false-halt benchmark passages
- PineconeBackend, WeaviateBackend, QdrantBackend vector store backends
- Bandit + Semgrep SAST in CI
- OTel spans enriched with h_logical, h_factual, warning, token_count

## v2.0.0

### Done
- Case-sensitivity fix in GroundTruthStore
- LLM judge error handling hardened
- SafetyKernel hard_limit validation
- Thread-safe OTel setup
- Named constants for LLM judge blending
- Documentation fixes (case-studies.md, changelog sync, BibTeX)
- Repo hygiene (.editorconfig, pre-commit, py.typed, Dockerfile non-root)
- Histogram bucket_counts O(n log n) optimization
- 12 fragile inspect.getsource tests replaced with behavioral tests
- New test files: knowledge, kernel validation, ingest

## v2.2.0 (planned)

- Lite scorer (no-NLI fast path with learned heuristics)
- ONNX GPU batch inference optimization
- Multi-turn conversation tracking (session state across review calls)
- LLM critic ensemble (multi-model agreement scoring)
- Stable public API freeze (breaking changes require major version)

## v2.3.0 (planned)

- Plugin architecture for custom scorer backends
- gRPC transport option for server.py
- Multi-GPU sharding for NLI model inference
- Third-party security audit
