# Director-AI — Performance Benchmarks

Fixture note: `benchmarks/multilingual_corpus.jsonl` contains 200 deterministic
EU-market regression cases across 8 languages. It is validated by
`tools/validate_multilingual_corpus.py` and documented in
`docs-site/benchmarks/multilingual-corpus.md`.

FrontierFail note: `benchmarks/frontierfail_seed_packet.toml` and
`benchmarks/frontierfail_cases.jsonl` define a seed regression packet for
production-failure benchmarking. It is explicitly not an externally validated
benchmark score; `tools/validate_frontierfail_packet.py` enforces that boundary.

PINT note: `benchmarks/pint_replication_packet.toml` and
`benchmarks/pint_seed_cases.jsonl` define a prompt-injection replication packet.
It records the upstream adapter contract and synthetic smoke cases only; it is
not an official PINT score. `tools/validate_pint_replication_packet.py` enforces
that boundary.

## Scoring Tiers

Director-AI uses a 5-tier scoring pyramid. Each tier trades latency for accuracy.

| Tier | Backend | BA (AggreFact) | Latency | Params | Notes |
|------|---------|---------------|---------|--------|-------|
| 1 | Heuristic keywords | ~55% | <1ms | 0 | Regex patterns, no model |
| 2 | Rules engine | ~58% | <1ms | 0 | Configurable rule chains |
| 3 | Embedding similarity | ~65% | 5ms | varies | SentenceTransformers cosine |
| 4 | Distilled NLI (DeBERTa-v3-xsmall) | 49.8% (**FAIL**) | 27ms CPU | 70M | Insufficient capacity, needs re-training |
| 5 | Full NLI (FactCG-DeBERTa-v3-Large) | 75.8% | 574ms GPU | 400M | Reference model |

### Full NLI (Tier 5) — AggreFact Breakdown

Model: `yaxili96/FactCG-DeBERTa-v3-Large` at threshold 0.46.

| Dataset | Samples | BA |
|---------|---------|-----|
| AggreFact-CNN | 558 | 68.8% |
| AggreFact-XSum | 558 | 74.3% |
| ClaimVerify | 1,088 | 78.1% |
| ExpertQA | 3,702 | 59.1% |
| FactCheck-GPT | 1,566 | 73.0% |
| Lfqa | 1,911 | 86.4% |
| RAGTruth | 16,371 | 82.2% |
| Reveal | 1,710 | 89.1% |
| TofuEval-MediaS | 726 | 71.9% |
| TofuEval-MeetB | 772 | 74.3% |
| Wice | 358 | 76.9% |
| **Macro Average** | **29,320** | **75.8%** |

### Distilled NLI (Tier 4)

Model: `anulum/director-ai-nli-lite` (DeBERTa-v3-xsmall, 70M params).
Training: 10 epochs, KL divergence + hard label blend (alpha=0.3), T=1.5.

- ONNX INT8 latency: 27ms CPU
- Model size: 83MB ONNX
- Sanity checks (6/6 PASS):
  - Supported claims: P[support] = 0.84–0.87
  - Contradicted claims: P[contra] = 0.74–0.89
- **Full AggreFact evaluation: 49.8% BA** (29,320 samples, per-dataset optimal)
  - This is at random chance — the student model does not generalise beyond
    the 6 hand-picked sanity examples
  - Root cause: DeBERTa-v3-xsmall (70M) lacks capacity for the FactCG
    instruction template pattern. Larger student or more training needed.
  - **Status: NOT production-ready.** Use Tier 5 (full NLI) or Tier 3
    (embedding) until distillation is improved.

---

## RAG Backends

All RAG techniques are independently toggleable decorator backends.
Measured on a synthetic 10K-document knowledge base.

| Backend | Operation | Latency Overhead | Memory Overhead |
|---------|-----------|-----------------|-----------------|
| BM25 Hybrid (RRF) | Query-time | +2–5ms | +index size |
| Cross-encoder reranker | Query-time | +15–50ms | +model load |
| Parent-child chunking | Index + query | +1ms query | +parent storage |
| Adaptive retrieval routing | Query-time | <1ms (heuristic) | negligible |
| HyDE (pseudo-doc) | Query-time | +LLM call | negligible |
| Query decomposition | Query-time | +LLM call per sub-query | negligible |
| Contextual compression | Query-time | +1–2ms (keyword), +LLM (advanced) | negligible |
| Multi-vector | Index + query | +2–3ms query | +summary/title vectors |

### RAG Decorator Stack

```
base (Chroma/Pinecone/FAISS/...)
  → HybridBackend (BM25+dense, RRF)
    → RerankedBackend (cross-encoder)
      → ParentChildBackend (return parent from child match)
        → HyDEBackend (pseudo-doc generation before embed)
          → QueryDecompositionBackend (split → parallel retrieve → merge)
            → ContextualCompressionBackend (LLM compress results)
```

All layers are enabled via config fields (e.g., `parent_child_enabled: true`).

---

## Agentic Swarm

Performance characteristics for SwarmGuardian multi-agent monitoring.

| Operation | Latency | Thread-safe |
|-----------|---------|-------------|
| Agent registration | <0.1ms | Yes (Lock) |
| Handoff scoring (keyword) | <0.5ms | Yes |
| Handoff scoring (NLI) | +NLI latency | Yes |
| Cascade halt propagation | O(n) agents | Yes |
| Metrics query | <0.1ms | Yes (Lock) |

### Framework Adapters

| Framework | Integration Point | Overhead |
|-----------|------------------|----------|
| LangGraph | Conditional edge | <0.5ms per edge |
| CrewAI | Task output callback | <0.5ms per task |
| OpenAI Swarm | Function wrapper | <0.5ms per handoff |
| AutoGen | Group chat filter | <0.5ms per message |

---

## Configuration

| Feature | Config field | Default |
|---------|-------------|---------|
| Scorer backend | `scorer_backend` | `auto` (rust > onnx > deberta > lite) |
| Hardened mode | `hardened` | `false` |
| Dry-run mode | `dry_run` | `false` |
| Production mode | `production_mode` | `false` |
| Cost tracking | `cost_tracking_enabled` | `false` |

---

## Test Coverage

| Category | Test files | Tests |
|----------|-----------|-------|
| Core scoring | 8 | ~120 |
| RAG backends | 6 | ~140 |
| Agentic/swarm | 5 | ~100 |
| CLI commands | 3 | ~80 |
| Compliance | 2 | ~40 |
| Config | 2 | ~30 |
| **Total** | **26+** | **500+** |

## PII Regex Scanner — Python vs Rust

The moderation package exposes ``RegexPIIDetector`` with an
optional Rust fast-path backed by ``backfire_kernel.PiiScanner``.
Rust uses ``regex::RegexSet`` to pre-filter non-matching patterns
in a single pass, walking the input once regardless of pattern
count.

| corpus | size (B) | Python ms/call (median) | Rust ms/call (median) | speedup (×) |
| --- | --- | --- | --- | --- |
| `clean-1kb` | 1225 | 0.413 | 0.110 | 3.74 |
| `clean-10kb` | 12250 | 2.959 | 0.145 | 20.44 |
| `mixed-1kb` | 2152 | 0.724 | 0.213 | 3.40 |
| `mixed-10kb` | 21530 | 5.826 | 0.493 | 11.82 |

**Reproduce:** ``python -m benchmarks.pii_scanner_bench --rounds 500``.
Raw data lands in ``benchmarks/results/pii_scanner_bench.json``.
Measured on i5-11600K, Python 3.12.3, ``backfire_kernel`` release
build. Clean corpora hit the ``RegexSet`` prefilter hard — Rust
skips every pattern once it decides none can match — so the speedup
is largest on benign text, which dominates production traffic.
The mixed corpora show the worst case where every pattern hits and
the scanner must actually walk.

`PIIRedactor` uses this detector path by default and exposes
`redact_with_report()` for production audit metadata. Reports contain stable
replacement categories, offsets, detector names, scores, and aggregate counts,
but never serialise raw matched values. This keeps redaction telemetry
tenant-safe while allowing compliance dashboards to prove which categories were
masked.

## Edge/Mobile Runtime Evidence

`benchmarks.edge_mobile_evidence` records the current R14 edge/mobile state for
browser, Worker, embedded, and local low-latency deployments:

```bash
PYTHONPATH=src python -m benchmarks.edge_mobile_evidence
```

The packet checks the WASM release plan, WASM source/tests/example, ONNX and
quantisation contracts, Rust kernel source, deployment docs, and latency
benchmark scripts. It intentionally separates local-trial readiness from
release readiness. Release readiness still requires an actual `wasm-pack`
artefact, quantised ONNX model artefact, browser/Web Worker smoke evidence,
mobile or embedded-device smoke evidence, and package-publish evidence.

## Rust vs Python E2E Comparison (Published Reproducible Packet)

Use the dedicated E2E runner to benchmark Rust-accelerated and forced-Python
execution paths side by side across a fixed deterministic scenario suite.

Run:

```bash
PYTHONPATH=src uv run python -m benchmarks.rust_python_e2e_compare --iterations 200 --warmup 30
```

Outputs:

- JSON artifact:
  `benchmarks/results/rust_python_e2e_compare_<UTC_TIMESTAMP>.json`
- publishable Markdown comparison report:
  `benchmarks/results/rust_python_e2e_compare_<UTC_TIMESTAMP>.md`

Each report includes per-scenario median and p95 latency for both modes,
speedup ratio (`Py/Rust`), and checksum parity to prove result consistency.

## Full Software Benchmark Campaign

Run the consolidated benchmark campaign (quality, E2E, retrieval, load, latency,
and Rust/Python comparison) as one reproducible packet:

```bash
PYTHONPATH=src python -m benchmarks.full_benchmark_campaign
```

Outputs:

- `benchmarks/results/full_benchmark_campaign_<UTC_TIMESTAMP>.json`
- `benchmarks/results/full_benchmark_campaign_<UTC_TIMESTAMP>.md`

Use `--strict` to return non-zero when any case fails or times out.

## Provenance And KB Lineage Evidence

For the R9 provenance gate, generate an evidence packet that checks KB snapshot
Merkle-root evolution, tenant-scoped roots, signed-fact conflict reporting,
citation-fact inclusion proofs, HMAC provenance-chain verification, and tamper
detection:

```bash
PYTHONPATH=src python -m benchmarks.provenance_evidence --fact-count 4
```

Outputs:

- `benchmarks/results/provenance_evidence_<UTC_TIMESTAMP>.json`

Passing this packet is local evidence only. Closing the online KB evolution gate
still requires an operator-owned feedback loop, archived deployment evidence,
and sign-off that the exported lineage packet matches the live tenant KB.

## Conformal Routing Evidence

For the R10 conformal-routing gate, generate an evidence packet that checks
deterministic 95% split-conformal coverage and the production routing policy:
allow only when the upper risk bound is low, send uncertain outputs to human
review or a stronger model, and reject only when the lower risk bound is high:

```bash
PYTHONPATH=src python -m benchmarks.conformal_routing_evidence \
  --coverage 0.95 \
  --calibration-samples 80 \
  --validation-samples 40
```

Outputs:

- `benchmarks/results/conformal_routing_evidence_<UTC_TIMESTAMP>.json`

Passing this packet is local evidence only. Closing the conformal-routing gate
still requires representative domain calibration data, archived deployment
evidence, and operator sign-off that the selected human-review or escalation
route is live.

## Trajectory Rollback Evidence

For the R11 trajectory-safety gate, generate an evidence packet that checks
Monte-Carlo preflight action bands and native rollback-hook behaviour:

```bash
PYTHONPATH=src python -m benchmarks.trajectory_rollback_evidence \
  --simulations 4
```

Outputs:

- `benchmarks/results/trajectory_rollback_evidence_<UTC_TIMESTAMP>.json`

Passing this packet is local evidence only. Closing the trajectory rollback gate
still requires an operator-owned live undo backend, adversarial trajectory
stress testing against deployment traffic, and sign-off that rollback evidence
is attached to incident/change-management records.

## Multimodal Temporal Evidence

For the R12 multimodal and temporal consistency gate, generate an evidence
packet that checks image allow/halt paths, caption-grounding conflicts, video
frame temporal halts, and dependency-free hash-bag image/text execution:

```bash
PYTHONPATH=src python -m benchmarks.multimodal_temporal_evidence
```

Outputs:

- `benchmarks/results/multimodal_temporal_evidence_<UTC_TIMESTAMP>.json`

Passing this packet is local evidence only. Closing the multimodal gate still
requires external Vision-NLI or equivalent benchmark evidence, real video/frame
model validation, and operator sign-off for deployment-specific modality
coverage.

## Federated Privacy Evidence

For the R13 privacy-preserving federation gate, generate an evidence packet that
checks DP-noised safety-signal aggregation, tenant/category contribution caps,
minimum cohort enforcement, and additive secret-sharing aggregate
reconstruction:

```bash
PYTHONPATH=src python -m benchmarks.federated_privacy_evidence
```

Outputs:

- `benchmarks/results/federated_privacy_evidence_<UTC_TIMESTAMP>.json`

Passing this packet is local evidence only. Closing the federated privacy gate
still requires an external federation run, malicious-secure aggregation review,
and deployment-specific poisoning-resilience evidence.

## Auto-Redteam Defence Evidence

For the R15 auto-redteam and defence-genome gate, generate an evidence packet
that checks two reviewed adversarial-mining cycles, detection-uplift gates,
registry promotions, and tenant-safe serialisation:

```bash
PYTHONPATH=src python -m benchmarks.auto_redteam_defence_evidence
```

Outputs:

- `benchmarks/results/auto_redteam_defence_evidence_<UTC_TIMESTAMP>.json`

Passing this packet is local evidence only. Closing the auto-redteam gate still
requires a live nightly run, operator-owned patch/model integration sign-off,
and external adversarial corpus evidence.

## Formal Symbolic Evidence

For the R16 formal and symbolic guard gate, generate an evidence packet that
checks DPLL formula halts/allows, the Lean runner contract, Z3 profile handling,
code-contract ordering, and tenant-safe serialisation:

```bash
PYTHONPATH=src python -m benchmarks.formal_symbolic_evidence
```

Outputs:

- `benchmarks/results/formal_symbolic_evidence_<UTC_TIMESTAMP>.json`

Passing this packet is local evidence only. Closing the formal-symbolic gate
still requires an external Lean proof run, an actual Z3 packet under `[formal]`
in release evidence, and operator-owned math/code/numeric domain contracts.

## Sustained Load Hardening Evidence

For the R17 production-hardening gate, generate an evidence packet that checks
async stream ordering under concurrent scheduling and same-key tenant poisoning
isolation:

```bash
PYTHONPATH=src python -m benchmarks.sustained_load_evidence \
  --streams 64 \
  --tokens-per-stream 64 \
  --tenant-cases 64
```

Outputs:

- `benchmarks/results/sustained_load_evidence_<UTC_TIMESTAMP>.json`

Passing this packet is local evidence only. Closing the production deployment
gate still requires a sustained staging or production run with external
telemetry, environment details, and security-review sign-off.

---

## Runtime Scorer Model Choices

Runtime scorer choices are separate from managed-training base models. Training
base models define what can be fine-tuned; runtime scorer choices define what a
deployment may offer for live NLI scoring after the Vertex benchmark gate.

Final Vertex report:

```text
gs://gotm-director-ai-training/managed-training/benchmarks/20260429T1409-e9859f8-full-e1-20260429-alias/model_benchmark_report.json
```

| Alias | Runtime source | Status | BA | F1 | Regression | Recommendation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `balanced-default` | managed FactCG DeBERTa v3 large artefact | stable | 0.752 | 0.7916 | -0.6 pp | deploy |
| `deberta-small` | managed DeBERTa v3 small artefact | stable | 0.747 | 0.7843 | -1.1 pp | deploy |
| `deberta-large-nli` | managed DeBERTa v3 large NLI artefact | stable | 0.740 | 0.7822 | -1.8 pp | deploy |
| `distilroberta-fast` | managed DistilRoBERTa artefact | domain only | 0.719 | 0.7604 | -3.9 pp | deploy domain only |
| `roberta-mnli-legacy` | managed RoBERTa MNLI artefact | domain only | 0.706 | 0.7529 | -5.2 pp | deploy domain only |

Stable options are exposed by default through `GET /v1/scorer/models`.
Domain-only choices require `DIRECTOR_ALLOW_DOMAIN_ONLY_SCORER_MODEL=true`.
Custom model IDs or paths require `DIRECTOR_ALLOW_CUSTOM_SCORER_MODEL=true`
and should pass the Vertex benchmark gate before production use.

### Per-model benchmark packages

Selectable scorer models are advertised only when they have a package
entry in `benchmarks/model_benchmark_packages.toml`. The package
manifest expands the stable runtime scorer aliases into the required
evidence queue:

- existing model-choice general gate,
- AggreFact anchor,
- RAGTruth,
- HaluEval,
- FinanceBench,
- ContractNLI/CUAD,
- MedNLI/PubMedQA,
- PatronusAI/HaluBench text benchmark.

Validate the package plan and inspect the next missing evidence item:

```bash
PYTHONPATH=src python -m benchmarks.model_benchmark_packages --root .
PYTHONPATH=src python -m benchmarks.model_benchmark_packages --root . --json
```

Vertex package execution preserves each package's scorer template in
`DIRECTOR_SCORER_TEMPLATE` and passes the same value to
`benchmarks.aggrefact_eval --scorer-template`. This is required for
managed FactCG artefacts because Vertex resolves GCS model URIs to local
cache paths, where automatic model-name detection cannot infer the
FactCG instruction template reliably. AggreFact package stages fail
closed when the result file is missing, when `balanced-default` drops
below `0.700` global balanced accuracy, or when predictions collapse
above a `0.950` majority share.

The current stable packages are `pending_external_suite`: the managed
general model-choice gate exists, but external benchmark packets still
need to be generated per model before model-specific public claims are
promoted. Domain-only models stay opt-in until their domain packet is
complete and reviewed.

---

*Generated from verified benchmark runs. Numbers are from local evaluation
on the mining rig (i5-11600K, GTX 1060 6GB / 5x RX 6600 XT 8GB).
AggreFact results are on the full 29,320-sample LLM-AggreFact benchmark.*
