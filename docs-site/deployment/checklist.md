# Production Checklist

Before deploying Director-AI to production, verify each item:

## Scoring

- [ ] Set `use_nli=True` with a production NLI model (FactCG-DeBERTa-v3-Large recommended)
- [ ] Configure `ground_truth_store` with your domain knowledge base
- [ ] Set appropriate `threshold` per domain — see [Domain Recommendations](../guide/threshold-tuning.md#domain-recommendation-table)
- [ ] Enable `cache_size > 0` for repeated queries (reduces NLI inference cost)

## Observability

- [ ] Set up OpenTelemetry with `setup_otel()` for trace collection
- [ ] Enable `metrics_enabled=True` for Prometheus-compatible metrics
- [ ] Configure `log_json=True` for structured log aggregation

## Server

- [ ] Use `director-ai serve --workers N` for multi-worker deployment (N = CPU cores)
- [ ] Configure rate limiting: `rate_limit_rpm > 0` for public endpoints
- [ ] Set `api_keys` for authentication on exposed endpoints
- [ ] Set non-root user in Docker (already done in v2.0.0 Dockerfile)

## Benchmarking

- [ ] Run `director-ai bench` to establish baseline metrics before deployment
- [ ] Run `director-ai bench --dataset e2e` to verify catch rate on your domain data
- [ ] Review `director-ai config --profile <domain>` settings match your requirements

## Security

- [ ] Audit dependencies: `pip-audit --strict`
- [ ] Run SAST: `bandit -r src/director_ai/`
- [ ] Review CORS origins — default is empty; public browser deployments need exact `https://` origins
- [ ] Ensure caller credentials are not logged or exposed in responses

## Recovery

- [ ] Create a git bundle backup before release or incident-response changes.
- [ ] Verify the backup checksum file.
- [ ] Restore-test the bundle with `tools/verify_repository_backup.py`.
- [ ] Store the JSON verification report with the release or incident evidence.
