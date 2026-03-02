# Roadmap

## v2.0.0 (current)

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

## v2.1.0 (planned)

- Multi-turn conversation tracking (session state across review calls)
- LLM critic ensemble (multi-model agreement scoring)
- RAG backend plugins (Pinecone, Weaviate, Qdrant)
- Stable public API freeze (breaking changes require major version)

## v2.2.0 (planned)

- Plugin architecture for custom scorer backends
- gRPC transport option for server.py
- Multi-GPU sharding for NLI model inference
- Third-party security audit
