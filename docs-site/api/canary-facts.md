# Counterfactual Canary Facts

A canary is a deliberately false, uniquely-marked fact planted in a tenant's
knowledge base. No legitimate answer should ever contain its sentinel token —
so a token that surfaces in a model's output, or a canary chunk that turns up in
the retrieved evidence, is direct evidence of **leakage, injection, or KB
poisoning**. Canaries are honeytokens for a RAG system.

## Two signals

- **leakage** — a canary's sentinel token appears in the model's answer. No
  grounded answer should reproduce it; its presence means exfiltration,
  injection, or a poisoned chunk being regurgitated.
- **citation** — a canary chunk appears in the evidence the answer was grounded
  in (recognised by its `kb_canary` metadata). Retrieval surfaced a honeytoken
  it never should have — pairs naturally with the
  [evidence firewall](evidence-firewall.md), which produces the screened chunk
  set.

Each detection yields a `CanarySignal`, increments
`canary_signals_total{signal}`, and fires an optional alert callback.

## Through the guard

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard.from_profile("finance")

# Plant a tenant-scoped canary (added to the KB; token never appears legitimately)
fact = guard.plant_canary("acme")

# Later, scan a model answer (and optionally the retrieved evidence)
signals = guard.scan_canaries(model_answer, "acme", evidence=retrieved_chunks)
for signal in signals:
    alert(signal.signal, signal.canary_id)   # "leakage" or "citation"
```

Canaries are strictly tenant-scoped: a scan for one tenant never matches another
tenant's tokens, so a leak is attributable to the tenant whose data escaped.

## Direct use

```python
from director_ai.core.canary import CanaryRegistry, CanaryDetector

registry = CanaryRegistry()
fact = registry.mint("acme")            # mint a canary for a tenant
# Plant fact.text in the vector store with fact.metadata() so citation
# detection can recognise the chunk.

detector = CanaryDetector(registry, alert=page_on_call)
detector.scan_answer(answer, "acme")    # leakage signals
detector.scan_evidence(chunks, "acme")  # citation signals
```

The token factory and clock are injectable on `CanaryRegistry` for deterministic
tests; in production the token is a cryptographically random sentinel.
