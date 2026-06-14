# Hallucination economics

> **Status: cost-risk decision layer.** The economics selector turns guarding
> from a fixed cost into a per-request decision — it picks the action whose
> expected total cost is lowest, and reports the value that guarding adds. It is
> a routing aid that sits on top of the scorers; the scorers remain the ground
> truth on whether a specific answer is grounded.

Guarding every request with the heaviest scorer wastes money on safe requests;
guarding nothing lets expensive mistakes through. `HallucinationEconomics` makes
the choice per request. Each guard action has an operational **cost** (compute,
latency, dollars) and a **catch** rate — the probability it stops a hallucination
that would otherwise reach the user. A hallucination that *does* reach the user
has a business cost that depends on the domain. For a request whose hallucination
risk is `risk`:

```
expected_cost(action) = action.cost + risk · (1 - action.catch) · hallucination_cost
```

The cheapest action under that rule is the economically optimal one.

## The default action tiers

| Action | Cost | Catch | When it wins |
|---|---|---|---|
| `skip` | 0.00 | 0.00 | negligible risk or trivial stakes |
| `heuristic` | 0.01 | 0.55 | low risk, low stakes |
| `nli` | 0.20 | 0.90 | moderate risk or stakes |
| `escalate` | 1.00 | 0.97 | high risk and high stakes |
| `human_review` | 5.00 | 0.99 | critical stakes where residual risk dominates |

Costs are abstract units — use one unit for both action cost and
`hallucination_cost` (a latency budget, a compute price, or currency). The menu
is fully configurable; supply your own `GuardAction` list for a deployment's real
cost model.

## Usage

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard()
risk = guard.forecast("What was the patient's last potassium reading?").risk

# Medical domain: a wrong answer is expensive.
decision = guard.guard_economics(risk, hallucination_cost=100.0)
print(decision.action, decision.expected_cost, decision.value)
# 'escalate' 3.7 86.3   -> escalate; guarding avoids 86.3 of expected loss
```

The decision carries the chosen action, its expected cost, the per-action
`breakdown`, the residual risk after the action, and `value` — the expected loss
guarding avoids versus doing nothing — so guarding can be reported as a value
driver, not only a cost centre. `worth_guarding` is `False` when doing nothing is
already optimal.

## Measured behaviour

On the bundled labelled scenarios (`python -m benchmarks.hallucination_economics`):

| Metric | Value |
|---|---|
| Decision accuracy (matches the economic optimum) | 1.00 |
| Mean expected loss avoided, guarded scenarios | ~205 (abstract units) |
| Throughput | ~33,000 decisions/s |

These numbers come from the committed benchmark and
`benchmarks/results/hallucination_economics.json`; reproduce them with the command
above. The decision is exact arithmetic over the action menu, so there is no
scoring model and no polyglot kernel in this layer — the risk it consumes comes
from the forecaster or a scorer, which carry their own Rust fast paths.
