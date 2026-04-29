# Types & Dataclasses

Shared data types returned by scorer, agent, and streaming methods. All are frozen or standard dataclasses.

## CoherenceScore

The primary return type from `CoherenceScorer.review()` and `score()`.

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Composite coherence score (0.0–1.0) |
| `approved` | `bool` | Whether `score >= threshold` |
| `h_logical` | `float` | Logical divergence (NLI contradiction probability) |
| `h_factual` | `float` | Factual divergence (RAG deviation) |
| `warning` | `bool` | `True` if score is between `threshold` and `soft_limit` |
| `evidence` | `ScoringEvidence \| None` | Retrieved evidence and scoring details |
| `strict_mode_rejected` | `bool` | `True` if rejected because NLI was unavailable in strict mode |
| `cross_turn_divergence` | `float \| None` | Cross-turn NLI score (set when session context exists) |
| `verdict_confidence` | `float \| None` | Guardrail confidence in its own verdict [0, 1] (v3.10.0) |
| `nli_model_confidence` | `float \| None` | NLI softmax entropy-based confidence (v3.10.0) |
| `signal_agreement` | `float \| None` | Agreement between h_logical and h_factual [0, 1] (v3.10.0) |
| `contradiction_index` | `float \| None` | Cross-turn self-contradiction severity [0, 1] (v3.10.0) |
| `injection_risk` | `float \| None` | Intent-grounded injection risk [0, 1] (when detection enabled) |

```python
approved, score = scorer.review(query, response)

print(f"Score: {score.score:.3f}")
print(f"H_logical: {score.h_logical:.3f}")
print(f"H_factual: {score.h_factual:.3f}")
print(f"Warning: {score.warning}")

if score.evidence:
    for chunk in score.evidence.chunks:
        print(f"  Source: {chunk.text[:80]}")
```

---

## ReviewResult {: #reviewresult }

Return type from `CoherenceAgent.process()`.

| Field | Type | Description |
|-------|------|-------------|
| `output` | `str` | Best approved response (or fallback) |
| `coherence` | `CoherenceScore \| None` | Coherence score of the response |
| `halted` | `bool` | Whether safety kernel halted |
| `candidates_evaluated` | `int` | Number of candidates generated |
| `fallback_used` | `bool` | Whether a fallback was activated |
| `halt_evidence` | `HaltEvidence \| None` | Structured halt reason |
| `safety_events` | `tuple[SafetyEvent, ...]` | Tenant-safe hook decisions emitted during review |

---

## ScoringEvidence {: #scoringevidence }

Evidence collected during scoring — retrieved KB chunks, NLI details, and attribution.

| Field | Type | Description |
|-------|------|-------------|
| `chunks` | `list[EvidenceChunk]` | Top-K retrieved chunks |
| `nli_premise` | `str` | NLI premise text used |
| `nli_hypothesis` | `str` | NLI hypothesis text used |
| `nli_score` | `float` | Raw NLI divergence score |
| `chunk_scores` | `list[float] \| None` | Per-chunk NLI scores |
| `premise_chunk_count` | `int` | Number of premise chunks |
| `hypothesis_chunk_count` | `int` | Number of hypothesis chunks |
| `attributions` | `list[ClaimAttribution] \| None` | Per-claim source attribution |
| `token_count` | `int \| None` | NLI token consumption |
| `estimated_cost_usd` | `float \| None` | Estimated NLI inference cost |
| `claim_coverage` | `float \| None` | Fraction of claims supported by source |
| `per_claim_divergences` | `list[float] \| None` | Per-claim divergence scores |
| `claims` | `list[str] \| None` | Decomposed atomic claims |

---

## EvidenceChunk {: #evidencechunk }

A single chunk of retrieved evidence.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Chunk text content |
| `distance` | `float` | Similarity distance (lower = more relevant) |
| `source` | `str` | Source identifier (default `""`) |

---

## ClaimAttribution {: #claimattribution }

Maps a summary claim to its source sentence.

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | The atomic claim |
| `claim_index` | `int` | Index of the claim in the decomposed list |
| `source_sentence` | `str` | Best-matching source sentence |
| `source_index` | `int` | Index of the source sentence |
| `divergence` | `float` | NLI divergence score (lower = better support) |
| `supported` | `bool` | Whether the claim is supported |

---

## HaltEvidence {: #haltevidence }

Structured halt reason with evidence chunks.

| Field | Type | Description |
|-------|------|-------------|
| `reason` | `str` | Halt mechanism that triggered |
| `last_score` | `float` | Coherence score at halt point |
| `evidence_chunks` | `list[EvidenceChunk]` | Contradicting chunks |
| `nli_scores` | `list[float] \| None` | NLI scores at halt point |
| `suggested_action` | `str` | Recommended action (e.g., "retry with KB context") |
| `trace_attribution` | `HaltTraceAttribution \| None` | Fact source, retrieval path, scorer path, token offset, threshold, and halt-margin data |
| `counterfactual_diagnostic` | `CounterfactualHaltDiagnostic \| None` | Single-fact change diagnostic for the halt |

```python
if session.halt_evidence_structured:
    ev = session.halt_evidence_structured
    print(f"Reason: {ev.reason}")
    print(f"Score: {ev.last_score:.3f}")
    if ev.trace_attribution:
        print(f"Token: {ev.trace_attribution.token_offset}")
    if ev.counterfactual_diagnostic and ev.counterfactual_diagnostic.best_change:
        print(ev.counterfactual_diagnostic.best_change.fact_source)
    for chunk in ev.evidence_chunks:
        print(f"  {chunk.text[:80]} (distance={chunk.distance:.3f})")
```

## HaltTraceAttribution {: #halttraceattribution }

Trace coordinates for the halt decision.

| Field | Type | Description |
|-------|------|-------------|
| `fact_source` | `str` | Source label from the contradicting evidence chunks |
| `retrieval_path` | `str` | Retrieval path used to gather evidence for the halt |
| `scorer_path` | `str` | Scorer method that produced the structured review |
| `token_offset` | `int` | Token index where the halt condition triggered |
| `threshold` | `float \| None` | Limit that was crossed |
| `causal_contribution` | `float` | Absolute margin beyond the limit |

## SafetyEvent {: #safetyevent }

Tenant-safe halt or policy decision emitted by a runtime hook. The schema is
shared by streaming, containment, attestation, ontology, trajectory, and
cyber-physical paths so audit logs and trace tooling can read one record shape.

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `str` | Frozen schema id, currently `director.safety_event.v1` |
| `event_id` | `str` | Opaque event id |
| `timestamp` | `str` | RFC-3339 UTC timestamp |
| `request_id` | `str` | Request correlation id, if available |
| `tenant_id` | `str` | Tenant id, if available |
| `hook_id` | `str` | Stable hook id such as `streaming.kernel` |
| `hook_scope` | `str` | One of `streaming`, `containment`, `attestation`, `ontology`, `trajectory`, `cyber_physical`, `swarm`, or `agent` |
| `policy_decision` | `str` | One of `allow`, `warn`, `halt`, or `block` |
| `halt_reason` | `str` | Machine-readable reason |
| `threshold` | `float \| None` | Decision threshold, when relevant |
| `observed_score` | `float \| None` | Score that drove the decision, when relevant |
| `latency_ms` | `float \| None` | Hook latency |
| `evidence_refs` | `tuple[str, ...]` | Evidence identifiers only; no raw fact text |
| `tenant_safe_explanation` | `str` | Short operator-facing explanation safe for tenant logs |
| `trace_attribution` | `HaltTraceAttribution \| None` | Trace coordinates for halt decisions |
| `attributes` | `dict[str, str]` | Additive string metadata for hook-specific fields |

```python
from director_ai.core.safety_event import SafetyEvent

event = SafetyEvent.from_halt_evidence(
    evidence,
    hook_id="streaming.kernel",
    request_id="req-123",
    tenant_id="tenant-1",
)

payload = event.to_dict()
```

Runtime hook verdicts now carry this schema directly:

| Hook | Event field |
|------|-------------|
| Streaming | `TokenEvent.safety_event`, `StreamSession.safety_events` |
| Containment | `ContainmentVerdict.safety_event`, plus `ReviewResult.safety_events` when attached to `CoherenceAgent` |
| Cyber-physical | `GroundingVerdict.safety_event` |
| Attestation | `PassportVerdict.safety_event` |
| Ontology | `OntologyViolation.safety_event` |
| Trajectory | `PreflightVerdict.safety_event` |
| Swarm | `HandoffResult.safety_event` |

## CounterfactualHaltDiagnostic {: #counterfactualhaltdiagnostic }

Answer to "what single fact change would have prevented this halt?"

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | Diagnostic question that was answered |
| `observed_score` | `float` | Stream score at the halt point |
| `threshold` | `float` | Halt threshold used by the counterfactual graph |
| `best_change` | `CounterfactualFactChange \| None` | First single-fact branch that prevents the halt |
| `candidates` | `list[CounterfactualFactChange]` | Candidate single-fact branches |

## CounterfactualFactChange {: #counterfactualfactchange }

Candidate single-fact intervention.

| Field | Type | Description |
|-------|------|-------------|
| `fact_source` | `str` | Source label for the fact branch |
| `original_fact` | `str` | Retrieved fact text at halt time |
| `proposed_fact` | `str` | Proposed replacement fact from the generated claim |
| `required_score_delta` | `float` | Minimum score increase needed to prevent the halt |
| `prevented_halt` | `bool` | Whether this branch satisfies the halt invariant |

---

## InjectionResult {: #injectionresult }

Return type from `InjectionDetector.detect()` and `ProductionGuard.check_injection()`.

| Field | Type | Description |
|-------|------|-------------|
| `injection_detected` | `bool` | Whether combined score exceeds threshold |
| `injection_risk` | `float` | Aggregated injection risk [0, 1] |
| `intent_coverage` | `float` | Fraction of grounded claims |
| `total_claims` | `int` | Total decomposed claims |
| `grounded_claims` | `int` | Claims aligned with intent |
| `drifted_claims` | `int` | Claims deviating from intent |
| `injected_claims` | `int` | Claims with no traceability to intent |
| `claims` | `list[InjectedClaim]` | Per-claim breakdown |
| `input_sanitizer_score` | `float` | Stage 1 pattern-match score |
| `combined_score` | `float` | Weighted Stage 1 + Stage 2 |

---

## InjectedClaim {: #injectedclaim }

Per-claim injection attribution from `InjectionDetector`.

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | The atomic claim text |
| `claim_index` | `int` | Position in decomposed list |
| `intent_divergence` | `float` | Forward NLI: intent → claim |
| `reverse_divergence` | `float` | Reverse NLI: claim → intent |
| `bidirectional_divergence` | `float` | min(forward, reverse) |
| `traceability` | `float` | Content-word overlap with intent |
| `entity_match` | `float` | Named-entity overlap with intent |
| `verdict` | `str` | `"grounded"`, `"drifted"`, or `"injected"` |
| `confidence` | `float` | Verdict confidence [0, 1] |

---

## Full API

::: director_ai.core.types.CoherenceScore

::: director_ai.core.types.ReviewResult

::: director_ai.core.types.ScoringEvidence

::: director_ai.core.types.EvidenceChunk

::: director_ai.core.types.ClaimAttribution

::: director_ai.core.types.HaltEvidence

::: director_ai.core.types.HaltTraceAttribution

::: director_ai.core.types.CounterfactualHaltDiagnostic

::: director_ai.core.types.CounterfactualFactChange
