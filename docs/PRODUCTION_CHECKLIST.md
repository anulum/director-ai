# Production Deployment Checklist

> **Module**: Director-AI | **Version**: 3.15.2 | **License**: GNU AGPL v3
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
director-ai quickstart --profile production
director-ai production-check --path director_guard
director-ai production-check --path director_guard --require-secrets

# OR, for manual integration:
director-ai wizard --cli    # interactive
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
hybrid_rrf_k: 60              # BM25 + dense Reciprocal Rank Fusion constant
reranker_enabled: true
retrieval_abstention_threshold: 0.3
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
- [ ] Enable OpenTelemetry where distributed traces are required; scorer spans
      cover cache lookup, retrieval, NLI, calibration, and judge escalation
- [ ] Export a tenant-safe operations packet with
      `build_observability_operations_report()` for halt forensics, drift
      alerts, readiness controls, and compliance evidence references
- [ ] Attach observability operations evidence to the Customer Model Factory
      release gate with the operations packet URI, dashboard evidence URI,
      drift review flag, readiness-control status, compliance-export status,
      and operator sign-off URI.

### Performance

- [ ] Use ONNX backend for GPU: `scorer_backend: onnx`
- [ ] Run `director-ai doctor` and resolve dependency or model-revision warnings
- [ ] Pre-warm model after `doctor` passes for the selected backend
- [ ] Set `cache_size: 1024` for repeated queries
- [ ] For high throughput: `director-ai stress-test --rps 100`
- [ ] Generate provenance and KB-lineage evidence:
      `PYTHONPATH=src python -m benchmarks.provenance_evidence --fact-count 4`
- [ ] Attach provenance lineage evidence to the Customer Model Factory release
      gate with the archived feedback-loop run URI, signed lineage packet URI,
      tenant KB snapshot URI, deployed-fact match flag, protected-claim conflict
      resolution flag, and operator sign-off URI.
- [ ] Generate conformal routing evidence:
      `PYTHONPATH=src python -m benchmarks.conformal_routing_evidence`
- [ ] Attach conformal routing evidence to the Customer Model Factory release
      gate with the representative domain calibration packet URI, deployment
      routing packet URI, target coverage, empirical coverage, calibration
      sample count, verified escalation route, reject-to-human availability,
      and operator sign-off URI.
- [ ] Generate trajectory rollback evidence:
      `PYTHONPATH=src python -m benchmarks.trajectory_rollback_evidence`
- [ ] Attach trajectory rollback evidence to the Customer Model Factory release
      gate with the simulation evidence URI, live undo backend URI, adversarial
      stress packet URI, incident/change-management record URI, verified
      rollback hook, idempotency check, tenant-safe audit check, and operator
      sign-off URI.
- [ ] Generate multimodal temporal evidence:
      `PYTHONPATH=src python -m benchmarks.multimodal_temporal_evidence`
- [ ] Attach multimodal temporal evidence to the Customer Model Factory release
      gate with the Vision-NLI or equivalent benchmark URI, real video/frame
      validation URI, modality coverage URI, image/audio/video-temporal guard
      verification, caption-grounding verification, and operator sign-off URI.
- [ ] Generate federated privacy evidence:
      `PYTHONPATH=src python -m benchmarks.federated_privacy_evidence`
- [ ] Attach federated privacy evidence to the Customer Model Factory release
      gate with the external federation run URI, malicious-secure aggregation
      review URI, poisoning-resilience packet URI, privacy budget ledger URI,
      DP aggregation, cohort gate, secret-sharing and contribution-cap
      verification flags, and operator sign-off URI.
- [ ] Generate auto-redteam defence evidence:
      `PYTHONPATH=src python -m benchmarks.auto_redteam_defence_evidence`
- [ ] Generate formal symbolic evidence:
      `PYTHONPATH=src python -m benchmarks.formal_symbolic_evidence`
- [ ] Generate edge/mobile runtime evidence:
      `PYTHONPATH=src python -m benchmarks.edge_mobile_evidence`
      and confirm `ready_for_release` only after WASM artefacts, quantised model
      artefacts, browser/Web Worker smoke, and mobile or embedded-device smoke
      evidence are attached.
- [ ] Attach edge/mobile evidence to the Customer Model Factory release gate
      with the edge runtime packet URI, quantised model artefact URI, WASM
      package evidence URI, browser Web Worker smoke URI, mobile or embedded
      smoke URI, package-publish evidence URI, latency profile URI, release
      verification flags, and operator sign-off URI.
- [ ] Validate the generated WASM package after `wasm-pack build`:
      `PYTHONPATH=src python tools/check_wasm_release_package.py`
- [ ] Run the browser Web Worker smoke:
      `PYTHONPATH=src python tools/run_wasm_browser_worker_smoke.py`
- [ ] Generate sustained async/tenant hardening evidence:
      `PYTHONPATH=src python -m benchmarks.sustained_load_evidence`
- [ ] Attach deployment-hardening evidence to the Customer Model Factory
      release gate with staging or production telemetry, the sustained-load
      packet URI, async-ordering and tenant-poisoning pass flags, and operator
      sign-off URI.
- [ ] Run the local release evidence gate:
      `PYTHONPATH=src python tools/check_local_release_evidence.py --root . --mode local`
- [ ] Before a customer release, run the strict release evidence gate:
      `PYTHONPATH=src python tools/check_local_release_evidence.py --root . --mode release`
      and resolve every release blocker.
- [ ] For CI or release dashboards, emit the strict gate as JSON:
      `PYTHONPATH=src python tools/check_local_release_evidence.py --root . --mode release --format json`

### Release documentation sync

Run this checklist whenever a release candidate changes safety hooks,
benchmarks, public defaults, or deployment boundaries:

- [ ] `CHANGELOG.md` names the changed runtime surface, public default, or
  benchmark runner.
- [ ] `ROADMAP.md` marks only shipped work as complete and keeps open external
  audit or validation work unchecked.
- [ ] `ARCHITECTURE.md` matches current module ownership, hook wiring, optional
  runtime boundaries, and Rust acceleration paths.
- [ ] `SECURITY.md` covers any new residual risk, public endpoint exposure,
  dependency surface, or tenant boundary.
- [ ] `VALIDATION.md` includes the latest public benchmark or adversarial
  validation update when public claims changed.
- [ ] `docs-site/` pages that link to those files still point at the same source
  of truth.
- [ ] GitHub App installation-token compatibility check passes:
  `python tools/validate_github_app_token_compat.py` (no fixed-length `ghs_`
  assumptions; stateless token format accepted).
- [ ] If this release mints GitHub App installation access tokens directly,
  pre-rollout verification is executed against both formats on
  `POST /app/installations/:installation_id/access_tokens`:
  `X-GitHub-Stateless-S2S-Token: enabled` and
  `X-GitHub-Stateless-S2S-Token: disabled`.

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
For deployment gates, pair Grafana with the tenant-safe operations packet from
`director_ai.ui.build_observability_operations_report()` so halt forensics,
drift alerts, readiness controls, and Article 15/SOC 2 evidence references are
reviewable without exposing raw prompts or responses.

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
