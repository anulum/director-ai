# Temporal Consistency Graph

Track the **structured claims** a system makes over time and flag the moment it
contradicts itself across sessions and documents — the "patient has diabetes on
Monday, patient does not have diabetes on Tuesday" class of failure that matters
for medical, legal, and financial workflows.

## How it differs from cross-document memory

[`CrossDocumentConsistencyMemory`](cross-document-memory.md) compares whole
documents by text similarity. The temporal consistency graph is complementary
and operates on *structured* assertions on a timeline: a claim is a tuple
`(subject, predicate, value, polarity)` recorded at a timestamp, in a session,
from a document. That structure lets it detect exact, auditable contradictions
rather than fuzzy textual overlap.

## Contradiction kinds

| Kind | When |
|------|------|
| `polarity` | The same `(subject, predicate, value)` is asserted, then negated (or vice versa) |
| `functional_value` | A predicate declared single-valued is asserted with two different values (e.g. one diagnosis per patient) |

Both are tenant-scoped: claims from different tenants never contradict each
other.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.temporal_consistency import TemporalClaim

guard = ProductionGuard(DirectorConfig())
graph = guard.temporal_consistency  # persists across sessions on this guard

graph.record(TemporalClaim(
    subject="patient:123", predicate="has_condition", value="diabetes",
    polarity=True, timestamp=monday_ts, session_id="mon",
))

contradictions = graph.record(TemporalClaim(
    subject="patient:123", predicate="has_condition", value="diabetes",
    polarity=False, timestamp=tuesday_ts, session_id="tue",
))

assert contradictions[0].kind == "polarity"
assert contradictions[0].earlier.session_id == "mon"
assert contradictions[0].later.session_id == "tue"
```

`record()` returns the contradictions introduced by that claim (one per
conflicting prior claim, ordered `earlier` → `later` by timestamp) and retains
them for later querying.

## Single-valued (functional) predicates

Declare predicates that may hold only one value per subject to catch silent value
changes:

```python
from director_ai.core.temporal_consistency import TemporalConsistencyGraph

graph = TemporalConsistencyGraph(functional_predicates={"diagnosis"})
graph.record(TemporalClaim("patient:1", "diagnosis", value="diabetes", timestamp=1.0))
flip = graph.record(TemporalClaim("patient:1", "diagnosis", value="healthy", timestamp=2.0))
assert flip[0].kind == "functional_value"
```

## Querying and audit

```python
graph.history("patient:123", "has_condition")     # claims oldest-first
graph.contradictions(tenant_id="acme")            # contradictions for a tenant
graph.report(tenant_id="acme")                     # tenant-safe summary dict
```

`report()` returns `{tenant_id, subjects, claim_count, contradiction_count,
contradictions, consistent}`. Like every serialisation here it is **tenant-safe**
— the raw `source_text` of a claim is omitted unless `include_text=True` is
passed.

## Tenant and privacy model

- claims are grouped and compared only within `(tenant_id, subject, predicate)`;
- tenant ids are validated against `^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$` (empty =
  the default tenant);
- `delete_tenant()` removes a tenant's claims and contradictions for
  right-to-delete workflows;
- comparison cost per recorded claim is linear in the size of its
  subject/predicate group.

The graph is in-memory and persists for the lifetime of the owning object; pair
it with the provenance ledger when a durable, tamper-evident record is required.
