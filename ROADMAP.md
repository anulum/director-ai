# Roadmap

## v1.9.0 (current)

- Soft-halt mode for StreamingKernel (sentence-boundary halt)
- JSON structured logging (`log_json` flag wired)
- OpenTelemetry optional integration
- Request ID propagation
- 100-passage false-halt benchmark
- Coverage threshold raised to 80%
- API reference + domain presets + monitoring docs
- Dependency pin tightening

## v2.0.0 (planned)

- Multi-turn conversation tracking (session state across review calls)
- LLM critic ensemble (multi-model agreement scoring)
- RAG backend plugins (Pinecone, Weaviate, Qdrant)
- Stable public API freeze (breaking changes require major version)
- Plugin architecture for custom scorer backends
- gRPC transport option for `server.py`
- Multi-GPU sharding for NLI model inference
- Third-party security audit
