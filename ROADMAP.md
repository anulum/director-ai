# Roadmap

## v1.8.0 (planned)

- ONNX Runtime GPU provider auto-selection (CUDA → TensorRT → CPU fallback)
- `DirectorConfig.from_yaml()` / `from_env()` for production deployment
- Async streaming integration tests with real LLM backends
- OpenTelemetry trace propagation through `CoherenceAgent` pipeline

## v1.9.0 (planned)

- Batched NLI for multi-document scoring in a single GPU pass
- Token-level confidence calibration (Platt scaling on held-out AggreFact split)
- Policy engine: custom Python callables alongside regex rules
- Grafana dashboard template for Prometheus metrics

## v2.0.0 (planned)

- Stable public API freeze (breaking changes require major version)
- Plugin architecture for custom scorer backends
- gRPC transport option for `server.py`
- Multi-GPU sharding for NLI model inference
- Third-party security audit
