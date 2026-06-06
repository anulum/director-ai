# Evidence Firewall

The evidence firewall screens every retrieved chunk **before the model sees
it**. Grounding only helps if the chunks reaching the model are the ones the
tenant is allowed to read, came from a verified write, are still fresh, and
carry facts rather than injected instructions. The firewall enforces that as an
admission gate in front of retrieval: each chunk runs ten checks and is either
admitted into the grounding context or quarantined with a stable, tenant-safe
reason code.

It is side-effect-free apart from counter metrics, deterministic for a given
clock, and opt-in — retrieval behaves exactly as before unless a deployment
enables it.

## The ten admission checks

| Check | Fails when | Reason code |
|---|---|---|
| Tenant authorisation | the chunk is owned by a tenant the request may not read | `tenant_mismatch` |
| Provenance present | no content digest, version, or signature marker exists | `provenance_missing` |
| Signature verified | the write was not signature-verified at ingest | `signature_unverified` |
| Content-hash match | a recorded text digest no longer matches the text | `content_hash_mismatch` |
| Expiry | the recorded `expires_at` has passed | `expired` |
| Age | the chunk is older than the policy's max age | `too_old` |
| Source owner | no source owner/key is recorded | `source_owner_unknown` |
| Sensitivity | the label is not in the allowed set | `sensitivity_blocked` |
| Allowed use case | the chunk's use-case list excludes the request | `use_case_not_allowed` |
| Poisoning scan | the text reads as an injected instruction | `poisoning_detected` |

A chunk owned by an empty tenant denotes a shared, non-tenant corpus and passes
the tenant check. A chunk with no recorded text digest passes the content-hash
check — absence of a digest is the provenance check's job, not the integrity
check's. A chunk with no allowed-use-case list is unrestricted.

The reason codes are drawn from a closed vocabulary and never contain chunk
text, so a quarantine decision can be logged and shipped to a customer audit
trail without leaking another tenant's data.

## Policy

`FirewallPolicy` selects which checks are enforced and with what bounds.
Defaults are fail-closed on the integrity-critical checks (tenant, provenance,
signature, content hash, expiry, poisoning) and opt-in on the corpus-shape
checks (sensitivity labels, declared use case, source owner) that depend on a
customer taxonomy.

```python
from director_ai.core.evidence_firewall import (
    EvidenceFirewall,
    FirewallContext,
    FirewallPolicy,
)

firewall = EvidenceFirewall(FirewallPolicy(max_age_seconds=90 * 86_400))

report = firewall.screen(results, FirewallContext(tenant_id="acme", now_unix=now))

for chunk in report.admitted:        # safe to hand to the model
    ...
for verdict in report.quarantined:   # held back, with reasons
    print(verdict.chunk.chunk_id, verdict.failed_reasons)
```

`FirewallPolicy.permissive()` is an explicit, named "firewall disabled" posture
for non-tenant development corpora; production should never use it.

The poisoning scan is dependency-free and runs on the hot path. It scores how
strongly a chunk's text reads as an instruction aimed at the model — *"ignore
the previous instructions"*, *"you are now …"*, a leaked system-prompt
fragment, or an embedded tool-call literal — rather than a fact aimed at the
user. The model-backed `InjectionDetector` can be injected in its place:

```python
firewall = EvidenceFirewall(policy, poison_scan=detector.injection_score)
```

## Wiring into retrieval

`VectorGroundTruthStore` takes an optional `evidence_firewall`. When supplied,
every retrieval batch is screened inside the active-results filter, so
quarantined chunks never reach `retrieve_context` or
`retrieve_context_with_chunks`. When omitted, retrieval is unchanged.

```python
from director_ai.core.evidence_firewall import EvidenceFirewall
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

store = VectorGroundTruthStore(evidence_firewall=EvidenceFirewall())
```

`DirectorConfig` builds and attaches the firewall automatically from
`evidence_firewall_*` settings. The firewall is off by default; set
`evidence_firewall_enabled=True` to turn it on:

```python
from director_ai.core.config import DirectorConfig

config = DirectorConfig(
    evidence_firewall_enabled=True,
    evidence_firewall_max_age_seconds=90 * 86_400,
    evidence_firewall_enforce_sensitivity=True,
    evidence_firewall_allowed_sensitivity=("public", "internal"),
)
store = config.build_store()         # firewall attached
```

## Report shape

`FirewallReport.to_dict()` and `ChunkVerdict.to_dict()` serialise to JSON-safe,
tenant-safe dicts — chunk ids, per-check outcomes, and reason codes, never raw
text — suitable for an audit record:

```json
{
  "admitted_count": 1,
  "quarantined_count": 1,
  "verdicts": [
    {"chunk_id": "doc1", "admitted": true, "checks": [...], "failed_reasons": []},
    {"chunk_id": "doc2", "admitted": false,
     "checks": [...], "failed_reasons": ["signature_unverified"]}
  ]
}
```

## Metrics

- `evidence_firewall_chunks_screened_total` — every chunk the firewall saw.
- `evidence_firewall_chunks_quarantined_total{reason}` — quarantines, labelled
  by reason code, so a dashboard can show which check is dropping chunks.

## Performance

Screening is branching plus a native `hashlib` SHA-256 recompute, so there is
no Rust path to add — the digest is already a C path and the rest is control
flow. `benchmarks/evidence_firewall.py` reports per-chunk screening latency and
the poison-scan share of it; on the reference workstation a chunk screens in
roughly 12 µs, of which the poison scan is about a third.
