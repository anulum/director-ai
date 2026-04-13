# Production Deployment Checklist

> **Module**: Director-AI | **Version**: 3.13.0 | **License**: GNU AGPL v3
>
> © Concepts 1996–2026 Miroslav Šotek. All rights reserved.

15-minute guide from zero to production-ready Director-AI deployment.

---

## 1. Choose Your Tier (2 min)

| Tier | Install | Accuracy | Latency | Use Case |
|------|---------|----------|---------|----------|
| **Rules** | `pip install director-ai` | Rule-based | <1 ms | Input validation, cheap filter |
| **Embed** | `pip install "director-ai[embed]"` | ~65% BA | 3 ms | Fast similarity check |
| **NLI** | `pip install "director-ai[nli]"` | 75.6% BA | 14.6 ms | Production fact-checking |
| **NLI+RAG** | `pip install "director-ai[nli,vector]"` | 75.6%+ | 20 ms | KB-grounded verification |
| **Full** | `pip install "director-ai[nli,vector,server]"` | 75.6%+ | 20 ms | Self-hosted API |

**Recommendation**: Start with `[nli]` for accuracy. Add `[vector]` when you have a knowledge base.

---

## 2. Minimal Integration (3 min)

### Option A: SDK Wrap (6 lines)

```python
from director_ai import guard
from openai import OpenAI

client = guard(
    OpenAI(),
    facts={"policy": "Refunds within 30 days only"},
)
# All completions are now scored. Hallucinations raise HallucinationError.
```

### Option B: One-shot Score (4 lines)

```python
from director_ai import score

result = score("What is the refund policy?", response_text,
               facts={"refund": "30 days"}, threshold=0.3)
```

### Option C: FastAPI Middleware (3 lines)

```python
from director_ai.integrations.fastapi_guard import DirectorGuard

app.add_middleware(DirectorGuard, facts={"policy": "30 days"}, on_fail="reject")
```

---

## 3. Configure for Production (5 min)

### Generate config

```bash
director-ai wizard --cli    # interactive
# OR
director-ai config --profile thorough > config.yaml
```

### Key settings

```yaml
# config.yaml
production_mode: true          # enforces auth requirement
coherence_threshold: 0.6       # tune per domain (medical: 0.8, creative: 0.4)
use_nli: true
scorer_backend: deberta        # or "onnx" for GPU acceleration
dry_run: false                 # set true first to monitor false-positive rate

# Retrieval (if using KB)
hybrid_retrieval: true
reranker_enabled: true
parent_child_enabled: true     # better context from small chunks

# Security
injection_detection_enabled: true
sanitize_inputs: true
redact_pii: true

# Observability
metrics_enabled: true
log_level: INFO
log_json: true
```

### Environment variables

```bash
export DIRECTOR_API_KEYS='["sk-prod-xxx"]'
export DIRECTOR_COHERENCE_THRESHOLD=0.6
export DIRECTOR_USE_NLI=true
export DIRECTOR_PRODUCTION_MODE=true
```

---

## 4. Pre-launch Checklist

### Accuracy

- [ ] Run `director-ai eval --dataset regression` on your domain data
- [ ] Tune `coherence_threshold` per task type (use `adaptive_threshold_enabled: true`)
- [ ] Enable `dry_run: true` for 24h to measure false-positive rate
- [ ] Review halted outputs in audit log before going live

### Security

- [ ] Set `production_mode: true` (requires API keys)
- [ ] Enable `injection_detection_enabled: true`
- [ ] Enable `sanitize_inputs: true`
- [ ] Deploy behind TLS reverse proxy (never expose HTTP directly)
- [ ] Set `cors_origins` to your domains (not `*`)
- [ ] Review `SECURITY.md` residual risks section

### Observability

- [ ] Enable `metrics_enabled: true`
- [ ] Deploy Grafana dashboard from `deploy/observability/grafana-dashboard.json`
- [ ] Configure Prometheus alerts from `deploy/observability/prometheus-alerts.yml`
- [ ] Enable `audit_log_path` for compliance trail

### Performance

- [ ] Use ONNX backend for GPU: `scorer_backend: onnx`
- [ ] Pre-warm model: `director-ai doctor` verifies model loads
- [ ] Set `cache_size: 1024` for repeated queries
- [ ] For high throughput: `director-ai stress-test --rps 100`

---

## 5. Cost Estimation

| Backend | Tokens/req | Cost/1M reqs (self-hosted) | Latency |
|---------|-----------|---------------------------|---------|
| Rules | 0 | CHF 0 (CPU only) | <1 ms |
| Embed | 0 | CHF 0 (CPU) | 3 ms |
| NLI (CPU) | 0 | CHF 0 (CPU) | 120 ms |
| NLI (GPU) | 0 | ~CHF 50/mo (GPU rental) | 15 ms |
| LLM Judge | ~200 | ~CHF 5/1M (GPT-4o-mini) | 500 ms |

Director-AI itself has **zero per-request token cost** — NLI runs locally.
The only token cost is the optional LLM-as-judge escalation path.

---

## 6. Monitoring in Production

### Key metrics to watch

| Metric | Healthy | Alert Threshold |
|--------|---------|-----------------|
| Hallucination rate | <5% | >15% for 5 min |
| Review latency p95 | <50 ms | >500 ms |
| Streaming halts/min | <1 | >10 |
| Drift score | <0.1 | >0.2 for 15 min |
| KB query failures | <1% | >5% |

### Grafana dashboard

Pre-built at `deploy/observability/grafana-dashboard.json` with 9 panels.

---

## 7. Scaling

| Deployment | Requests/sec | Setup |
|-----------|-------------|-------|
| Single process | ~50 | `director-ai serve` |
| Uvicorn workers | ~200 | `--workers 4` |
| Docker + HPA | ~1000+ | `deploy/helm/director-ai/` |
| Cloud Run | Auto-scale | `deploy/cloud-run/Dockerfile.saas` |

---

## 8. Compliance (EU AI Act)

```bash
# Generate Article 15 report
director-ai compliance --days 30 --output report.html

# Or programmatically
from director_ai.compliance.reporter import ComplianceReporter
report = ComplianceReporter(audit_log).generate(days=30)
```

---

*Director-AI — ANULUM Institute | [anulum.li](https://www.anulum.li)*
