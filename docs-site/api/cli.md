# CLI Reference

Director-AI ships a command-line interface for scoring, serving, benchmarking, and project scaffolding.

```bash
pip install director-ai
director-ai --help
```

## Commands

### Scoring

```bash
# Score a single prompt/response pair
director-ai review "What is the capital of France?" "The capital is Berlin."

# Process with agent (generate + score)
director-ai process "What is the refund policy?"

# Batch score from JSONL
director-ai batch input.jsonl --output results.jsonl
```

### Server

```bash
# Start REST server (default transport: http)
director-ai serve --port 8080 --workers 4

# Start gRPC server
director-ai serve --transport grpc --port 50051 --workers 4

# Health check (via curl, no dedicated CLI command)
curl http://localhost:8080/v1/health
```

### Configuration

```bash
# Show current config
director-ai config

# Show a named profile
director-ai config --profile medical

# Generate YAML config
director-ai config --export config.yaml
```

### Project Scaffolding

```bash
# Create a new project with config, facts, and guard script
director-ai quickstart --profile medical
cd director_guard/
python guard.py
```

### Benchmarking

```bash
# Run latency benchmark
director-ai bench

# Run with specific dataset
director-ai bench --dataset e2e

# Run regression suite
python -m benchmarks.regression_suite
```

### Model Export

```bash
# Export to ONNX
director-ai export --format onnx --output ./models/onnx/

# Export to TensorRT
director-ai export --format tensorrt --output ./models/trt/
```

### Fine-Tuning

```bash
# Fine-tune NLI model on custom data
director-ai finetune train.jsonl --output ./models/custom/
```

### Threshold Tuning

```bash
# Adaptive threshold calibration on your dataset
director-ai tune eval_data.jsonl
```

### Version

```bash
director-ai version
# director-ai 3.10.0
```

## Global Options

| Flag | Description |
|------|-------------|
| `--config PATH` | YAML config file |
| `--profile NAME` | Named profile (fast, thorough, medical, etc.) |
| `--verbose` | Enable debug logging |
| `--json` | JSON output format |
