<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — Evaluation onboarding
-->

# Evaluation Onboarding

This guide gives evaluators, pilots, and new contributors a concrete path from
first install to evidence-backed deployment. Use it as the shared checklist for
a first Director-AI trial.

## Pilot Charter

Write this charter before tuning thresholds:

| Field | Question to answer |
|---|---|
| Workflow | Which exact model output will Director-AI inspect? |
| Consequence | What happens if this output is wrong? |
| Evidence source | Which governed facts, documents, or labelled examples define correctness? |
| Control action | Should the guard raise, annotate, halt, reject, or route to review? |
| Acceptance metric | Which false-positive, false-negative, latency, and review-cost limits matter? |
| Owner | Who approves threshold changes and production rollout? |

The first pilot is successful when this charter has measured evidence, not when
every optional integration has been enabled.

## 1. Choose The Pilot Shape

Pick one primary workflow before changing thresholds or adding integrations.

| Pilot shape | Best first artefact | Success signal |
|---|---|---|
| SDK wrapper | `guard()` around one existing client | Unsupported claims are blocked or annotated |
| RAG assistant | Vector-backed facts plus batch checks | Rejections cite the right source chunks |
| Streaming app | `StreamingKernel` or proxy stream guard | Bad partial output halts before completion |
| Internal review queue | Human review plus calibration store | Review decisions improve threshold selection |
| Enterprise gateway | REST/gRPC service behind auth and rate limits | Tenant-safe audit events are produced |

Keep the first pilot narrow. One workflow with good evidence is more useful
than many integrations with unclear acceptance criteria.

## 2. Install The Right Extras

| Need | Install |
|---|---|
| In-process scoring and SDK wrapping | `pip install director-ai` |
| NLI-backed factual scoring | `pip install "director-ai[nli]"` |
| RAG and vector stores | `pip install "director-ai[nli,vector]"` |
| REST service | `pip install "director-ai[nli,server]"` |
| Full local evaluation stack | `pip install -e ".[dev,nli,vector,server]"` |

Start without optional runtimes unless the pilot requires them. Rust, ONNX,
TensorRT, Go, Julia, Lean, gRPC, and WASM paths are deployment accelerators or
specialised proof surfaces, not required for the first evaluation.

## 3. Run A First Check

```python
from director_ai import score

result = score(
    "What is the refund policy?",
    "Refunds are available for 90 days.",
    facts={"policy": "Refunds are available for 30 days only."},
    threshold=0.3,
)

print(result.approved, result.score, result.evidence)
```

Expected outcome: the answer should be rejected or assigned a low coherence
score because it contradicts the governed fact.

## 4. Add Domain Evidence

For a real pilot, replace toy facts with governed data:

- product policy extracts;
- medical, legal, finance, or scientific reference snippets;
- customer-support macros;
- approved retrieval chunks from an internal knowledge base;
- labelled examples of acceptable and unacceptable answers.

Every benchmark or threshold decision should record:

- dataset source and ownership;
- split policy;
- scorer configuration;
- threshold;
- false-positive and false-negative examples;
- latency and hardware context;
- review owner and approval date.

## 5. Pick A Control Action

Director-AI can fail closed or fail soft. Choose based on consequence:

| Failure mode | Use when |
|---|---|
| `raise` | A wrong answer must not leave the application boundary |
| `metadata` | The application decides how to display or route the risk |
| `log` | Early observation without user-visible enforcement |
| HTTP reject | Proxy or middleware owns the response boundary |
| Human review | Low-confidence outputs need operator approval |
| Streaming halt | Partial output must stop before completion |

## 6. Validate The Pilot

Minimum evidence before a production pilot:

- quickstart or API smoke passes in the target environment;
- a labelled domain sample has been scored;
- at least one known bad answer is rejected;
- at least one known good answer is approved;
- logs contain no secrets;
- evidence records are understandable to a domain reviewer;
- operational ownership is assigned for threshold changes and false-positive
  review.

## 7. Evidence Packet Template

Use this packet for internal review, procurement, or a customer pilot closeout:

| Section | Contents |
|---|---|
| Use case | Workflow, users, output boundary, and consequence of a wrong answer |
| Dataset | Source, ownership, split policy, examples excluded from tuning |
| Configuration | Package version, scorer mode, threshold, runtime extras, KB version |
| Results | Good-answer approvals, bad-answer rejections, false positives, false negatives |
| Operations | Auth, tenant binding, metrics, logs, review queue, rollback owner |
| Claim boundary | Which claims are supported by this pilot and which require more evidence |

## 8. Continue By Role

| Role | Next page |
|---|---|
| Application developer | [Quickstart](../quickstart.md) |
| RAG engineer | [KB Ingestion](kb-ingestion.md) |
| Evaluation engineer | [BatchProcessor API](../api/batch.md) |
| Platform operator | [Production Guide](../deployment/production.md) |
| Compliance reviewer | [Compliance Reporting](compliance-reporting.md) |
| Commercial evaluator | [Product Overview](product-overview.md) |

## 9. Business Evidence Package

For commercial pilots, pair technical checks with a short evidence packet:

- one-page use-case statement (input, output, and consequence),
- fixed test split and acceptance criteria,
- false-positive and false-negative examples from production-like samples,
- incident and rollback flow,
- review owner and escalation path,
- and a short post-pilot report with expected reduction in factual-risk incidents.

Publishing this package early is often the fastest way to get procurement and
security teams aligned.
