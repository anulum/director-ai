# Customer Model Factory

Customer Model Factory packages customer-owned guardrail scorers from private
trace data through release-gated runtime configuration. The implementation is
designed for enterprise evidence review: every promotion decision is bound to
customer, workspace, tenant, deployment, artefact hash, and explicit blocker
records.

Customer-specific accuracy claims require package-specific benchmark evidence.
The factory can enforce zero silent unsafe passes as an objective profile, but
that is a scoped package control, not a universal accuracy claim.

## Operator Workflow

1. **Dataset validation** validates customer guardrail traces for required
   fields, split coverage, exact cross-split leakage, customer/tenant binding,
   references, severity, decisions, secrets redaction, and optional vertical
   metadata.
   Required scalar fields must be non-empty strings; blank split, severity,
   decision, identity, prompt, response, label, reviewer, or timestamp fields
   block readiness before training or benchmarking.
2. **Training manifest** binds a ready dataset report to immutable base-model
   provenance, training lane, hyperparameters, objective profile, and output
   artefact URI.
3. **Benchmark selection** records severity-aware metrics, raw result URI,
   claim boundary, selected benchmark, selected model artefact, and deterministic
   selection hash.
4. **Deployment manifest** binds the selected model to thresholds, abstention
   and escalation policy, audit log URI, evidence URI, rollback URI, retention,
   telemetry mode, and environment.
5. **Sector-extension boundary** records where customer-specific sector packs,
   database-class mappings, private retrieval schemas, tuning recipes, and
   customer benchmark packages attach without publishing those commercial
   assets in the public repository.
6. **Evidence pack** binds deployment, selection, selected benchmark, selected
   model, audit-log, rollback, telemetry, and reviewed sector artefacts into
   one deterministic export manifest.
7. **Runtime package** produces a customer-implementable private runtime
   configuration from the deployment and evidence pack.
8. **Monitoring manifest** records drift, false-positive review queue,
   false-negative incident queue, abstention rate, escalation rate, p95 latency,
   cost, package version, and retraining recommendations.
9. **Risk register** blocks accepted risks that lack an owner, expiry,
   compensating controls, or links to current evidence and monitoring hashes.
10. **Release gate** consumes enterprise readiness, runtime package, evidence
    pack, monitoring manifest, and risk register outputs. Promotion is allowed
    only when all blockers are absent.

## Customer Examples

The examples are local helpers. They load a generated runtime package and build
transport-neutral payloads without opening network connections:

```bash
python examples/customer_model_factory_runtime.py
python examples/customer_model_factory_rest_payload.py
```

The REST payload example builds a `POST /v1/score` payload that preserves
customer, workspace, tenant, deployment, model artefact, threshold, telemetry,
audit-log, evidence, and callback-policy fields from the runtime package.

## Operator CLIs

Generate a deterministic end-to-end fixture:

```bash
PYTHONPATH=src python tools/generate_customer_model_factory_fixture.py \
  --output-dir /tmp/director-ai-cmf-fixture
```

Assemble the final release gate from manifests:

```bash
PYTHONPATH=src python tools/assemble_customer_model_factory_release.py \
  --release-id release-customer-20260518 \
  --generated-at 2026-05-18T18:45:00Z \
  --enterprise-readiness /tmp/director-ai-cmf-fixture/enterprise_readiness.json \
  --runtime-package /tmp/director-ai-cmf-fixture/runtime_package.json \
  --evidence-pack /tmp/director-ai-cmf-fixture/evidence_pack.json \
  --monitoring-manifest /tmp/director-ai-cmf-fixture/monitoring_manifest.json \
  --risk-register /tmp/director-ai-cmf-fixture/risk_register.json \
  --observability-operations-evidence /tmp/director-ai-cmf-fixture/observability_operations_evidence.json \
  --provenance-lineage-evidence /tmp/director-ai-cmf-fixture/provenance_lineage_evidence.json \
  --conformal-routing-evidence /tmp/director-ai-cmf-fixture/conformal_routing_evidence.json \
  --trajectory-rollback-evidence /tmp/director-ai-cmf-fixture/trajectory_rollback_evidence.json \
  --multimodal-temporal-evidence /tmp/director-ai-cmf-fixture/multimodal_temporal_evidence.json \
  --federated-privacy-evidence /tmp/director-ai-cmf-fixture/federated_privacy_evidence.json \
  --edge-mobile-evidence /tmp/director-ai-cmf-fixture/edge_mobile_evidence.json \
  --auto-redteam-defence-evidence /tmp/director-ai-cmf-fixture/auto_redteam_defence_evidence.json \
  --formal-symbolic-evidence /tmp/director-ai-cmf-fixture/formal_symbolic_evidence.json \
  --deployment-hardening-evidence /tmp/director-ai-cmf-fixture/deployment_hardening_evidence.json \
  --output /tmp/director-ai-cmf-fixture/release_gate.json
```

The release gate fails closed unless observability evidence comes from staging
or production and includes the exported operations packet, dashboard evidence,
readiness controls, compliance exports, drift review, and operator sign-off.
KB provenance evidence must include an archived feedback-loop run, signed
lineage packet, tenant KB snapshot, deployed-fact match, conflict-resolution
status, and operator sign-off.
Conformal routing evidence must include representative domain calibration,
archived deployment routing evidence, target and empirical coverage, verified
human-review or stronger-model escalation, reject-to-human availability, and
operator sign-off.
Trajectory rollback evidence must include simulation evidence, live undo
backend evidence, adversarial stress evidence, incident/change-management
record linkage, verified rollback hook, idempotency check, tenant-safe audit
check, and operator sign-off.
Multimodal temporal evidence must include Vision-NLI or equivalent benchmark
evidence, real video/frame validation, deployment modality coverage, image,
audio, video-temporal, and caption-grounding verification flags, and operator
sign-off.
Federated privacy evidence must include an external federation run,
malicious-secure aggregation review, poisoning-resilience packet, privacy
budget ledger, DP aggregation, cohort gate, secret-sharing, and contribution-cap
verification flags, and operator sign-off.
Edge/mobile evidence must include the edge runtime packet, quantised model
artefact, WASM package evidence, browser Web Worker smoke, mobile or embedded
smoke, package-publish evidence, latency profile, verification flags for each
release control, and operator sign-off.
Auto-redteam defence evidence must include a live nightly run, defence update
packet, registry snapshot, external adversarial corpus, patch/model integration
sign-off, rollback plan, repeated-cycle, detection-uplift, registry-promotion,
tenant-safe report, and rollback verification flags, and operator sign-off.
Formal-symbolic evidence must include the local formal-symbolic packet,
external Lean proof run, actual Z3 release packet, operator-owned
math/code/numeric domain contracts, code-contract packet, DPLL guard
verification, Lean/Z3 verification, code-contract ordering, tenant-safe
serialisation, domain-contract verification, and operator sign-off.
Deployment-hardening evidence must likewise include archived telemetry, the
sustained-load packet, async-ordering and tenant-poisoning pass flags, and
operator sign-off.

Verify public documentation, examples, schemas, and CLI names remain aligned:

```bash
PYTHONPATH=src python tools/verify_customer_model_factory_docs_freeze.py --root .
```

Verify public open-core files do not expose proprietary sector-pack modules,
sector-specific metadata schemas, or private customer fixture identifiers:

```bash
PYTHONPATH=src python tools/verify_public_sector_boundary.py --root .
```

Verify implementation, tests, schemas, API docs, guide coverage, package
exports, and public docstrings remain aligned:

```bash
PYTHONPATH=src python tools/verify_customer_model_factory_compliance.py --root .
```

## Schema Surface

The public schema files live under `schemas/`:

- `customer-model-factory-trace.schema.json`
- `customer-model-factory-training-manifest.schema.json`
- `customer-model-factory-selection.schema.json`
- `customer-model-factory-deployment.schema.json`
- `customer-model-factory-sector-metadata.schema.json`
- `customer-model-factory-evidence-pack.schema.json`
- `customer-model-factory-runtime-package.schema.json`
- `customer-model-factory-monitoring.schema.json`
- `customer-model-factory-risk-register.schema.json`
- `customer-model-factory-release-gate.schema.json`

## Verification

The scoped Customer Model Factory verification suite covers the contract end to
end:

```bash
PYTHONPATH=src python -m pytest tests/test_customer_model_factory_*.py -q
PYTHONPATH=src python -m ruff check \
  src/director_ai/core/customer_model_factory \
  tests/test_customer_model_factory_*.py \
  tools/*customer_model_factory*.py \
  examples/customer_model_factory_*.py
PYTHONPATH=src python tools/verify_customer_model_factory_readiness.py --root .
PYTHONPATH=src python tools/verify_customer_model_factory_compliance.py --root .
PYTHONPATH=src python tools/verify_customer_model_factory_docs_freeze.py --root .
```
