# Evidence Packet (one-command demo)

The evidence packet is the single artefact a buyer or auditor runs to see the
whole guard loop work and to keep a verifiable record of it. It runs the narrow
grounding demo and seals the result so it can be checked later without re-running
the guard.

```bash
director-ai evidence --emit evidence/        # run the demo, write a sealed packet
director-ai verify-evidence evidence/        # re-check integrity + outcomes
```

## What it does

`director-ai evidence` runs the seven-step demo on a `ProductionGuard`:

1. Load a small policy knowledge base (`DEMO_FACTS`).
2. Ask a policy question.
3. Score a **grounded** answer — expected approved.
4. Score a **hallucinated** answer — expected blocked.
5. (Streaming halt is exercised by the streaming kernel demo.)
6. Emit, per decision, the [Answer Bill of Materials](answer-bom.md) and the
   [OpenTelemetry eval record](eval-trace.md).
7. Record the decisions in the packet (and, in a server deployment, the audit
   log).

The packet is written to `evidence_packet.json`:

```json
{
  "content": {
    "schema_version": "director.evidence_packet.v1",
    "knowledge_base_size": 5,
    "question": "What is the refund window?",
    "grounded": {"approved": true, "score": 0.92, "answer_bom": {...}, "eval_trace": {...}},
    "hallucinated": {"approved": false, "score": 0.2, "answer_bom": {...}, "eval_trace": {...}},
    "checks": {"grounded_approved": true, "hallucinated_blocked": true}
  },
  "integrity": {"algorithm": "sha256", "digest": "…"}
}
```

## Verification

`director-ai verify-evidence` (or `verify_evidence_packet`) recomputes the
SHA-256 digest over the canonical content and confirms the demo expectations —
grounded approved, hallucinated blocked. Any edit to the content changes the
digest, so tampering is caught:

```python
from director_ai.core.evidence_packet import (
    build_evidence_packet,
    verify_evidence_packet,
)
from director_ai.guard import ProductionGuard

packet = build_evidence_packet(ProductionGuard.from_profile("fast"))
ok, reason = verify_evidence_packet(packet)
```

Clear grounded-vs-hallucinated separation requires the model-backed scorer from
the `director-ai[nli]` extra; without it both answers score the heuristic
fallback and the demo expectations will not be met.
