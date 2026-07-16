# CLI Reference

Director-AI ships a command-line interface for scoring, serving, benchmarking, and project scaffolding.

```bash
pip install director-ai
director-ai --help
```

Top-level help is generated from the same command registry used by the
dispatcher, so every installed command listed below is discoverable through
`director-ai --help`.

## Commands

### Scoring

```bash
# Show core scoring and batch options without loading scorer/runtime paths
director-ai review --help
director-ai process --help
director-ai batch --help

# Score a single prompt/response pair
director-ai review "What is the capital of France?" "The capital is Berlin."

# Process with agent (generate + score)
director-ai process "What is the refund policy?"

# Batch score from JSONL
director-ai batch input.jsonl --output results.jsonl
```

### Ingestion

```bash
# Show ingest storage and chunking options without opening a path
director-ai ingest --help

# Ingest one file or a directory into an in-memory vector store
director-ai ingest ./knowledge

# Persist chunks for reuse across runs
director-ai ingest ./knowledge --persist ./chroma --chunk-size 350
```

### Server

```bash
# Show server transport and hardening options without starting the server
director-ai serve --help

# Start REST server (default transport: http)
director-ai serve --port 8080 --workers 4

# Start gRPC server
director-ai serve --transport grpc --port 50051 --workers 4

# Health check (via curl, no dedicated CLI command)
curl http://localhost:8080/v1/health
```

### Configuration

```bash
# Show configuration options without reading environment settings
director-ai config --help

# Show knowledge-base health options without opening the configured store
director-ai kb-health --help

# Show wizard and safety-dashboard options without launching UI dependencies
director-ai wizard --help
director-ai safety-dashboard --help

# Show current config
director-ai config

# Check runtime dependencies and model revision pins
director-ai doctor --help
director-ai doctor

# Show licence administration and deployment checks without reading licence files
director-ai license --help

# Show a named profile
director-ai config --profile medical

# Save a named profile view for editing
director-ai config --profile medical > config.yaml
```

### Project Scaffolding

```bash
# Show scaffold options without creating files
director-ai quickstart --help

# Create a new project with config, facts, and guard script
director-ai quickstart --profile medical
cd director_guard/
python guard.py

# Create and validate an authenticated production scaffold
director-ai quickstart --profile production
director-ai production-check --path director_guard
director-ai production-check --path director_guard --require-secrets
```

### Benchmarking

```bash
# Show benchmark command options without running benchmark work
director-ai eval --help
director-ai bench --help

# Show calibration and fine-tuning options without opening data files
director-ai tune --help
director-ai finetune --help

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

# Build a TensorRT engine cache from an existing ONNX export
director-ai export --format tensorrt \
  --onnx-dir ./models/onnx/ \
  --output ./models/onnx/trt_cache/
```

### Guardrail Forensics

```bash
# Show verification and diagnostics options without running scorer work
director-ai verify-numeric --help
director-ai verify-reasoning --help
director-ai temporal-freshness --help
director-ai check-step --help
director-ai consensus --help
director-ai adversarial-test --help

# Show KPI, forensics, and cost-report options without opening data/config
director-ai kpis --help
director-ai forensics --help
director-ai cost-report --help

# Explain reviewed misses from tenant-safe eval records
director-ai forensics --input eval_records.json --format markdown
```

The input is either a JSON array of eval records or an object with a `records`
array. It may include `director.eval.*` attributes from the eval-trace layer plus
reviewer labels such as `label: "hallucination"` or `label: "grounded"`.

### Fine-Tuning

```bash
# Fine-tune NLI model on custom data
director-ai finetune train.jsonl --output ./models/custom/
```

### Managed Training

Managed training submissions use one CLI contract across local, portable, and
Vertex execution lanes. `local` runs on the current machine. `portable` emits a
provider-neutral container job request for AWS, Azure, Slurm, Kubernetes, or
other customer-owned orchestrators. `vertex` submits directly to Vertex AI when
the `managed-training` extra and cloud credentials are installed.

```bash
# Local dry run
director-ai train submit \
  --backend local \
  --dataset-uri ./train.jsonl \
  --output-uri ./artifacts/customer-run-001 \
  --dry-run

# Portable external-orchestrator contract
director-ai train submit \
  --backend portable \
  --dataset-uri s3://customer-data/train.jsonl \
  --eval-uri azure://customer-data/eval.jsonl \
  --output-uri file:///mnt/customer-artifacts/director-ai/run-001 \
  --image registry.example.com/director-ai/train:2026-05 \
  --dry-run

# Vertex managed submission
director-ai train submit \
  --backend vertex \
  --dataset-uri gs://customer-data/train.jsonl \
  --eval-uri gs://customer-data/eval.jsonl \
  --output-uri gs://customer-artifacts/director-ai/run-001 \
  --project customer-project \
  --region europe-west4 \
  --image europe-west4-docker.pkg.dev/customer-project/director/train:2026-05
```

The portable backend is dry-run only by design. It redacts secret-looking
environment variables in the emitted request and leaves live job lifecycle
control to the customer's external orchestrator.

### Threshold Tuning

```bash
# Adaptive threshold calibration on your dataset
director-ai tune eval_data.jsonl
```

### Version

```bash
director-ai version
# director-ai 3.18.0
```

## Global Options

| Flag | Description |
|------|-------------|
| `--config PATH` | YAML config file |
| `--profile NAME` | Named profile (fast, thorough, medical, etc.) |
| `--verbose` | Enable debug logging |
| `--json` | JSON output format |
