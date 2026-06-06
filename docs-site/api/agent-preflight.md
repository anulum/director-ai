# Agent / MCP Preflight Guard

An agent that can call tools, hand off to other agents, and take real-world
actions needs a guard at the **seams**, not only on the final answer. The
preflight guard provides one evidence- and policy-tied decision per seam, so a
host loop (or an MCP server) can allow, warn, block, or escalate at each step.

Every hook returns a `PreflightDecision` whose `decision` is from a closed
vocabulary (`allow` / `warn` / `block` / `escalate`) and whose `reason` is a
tenant-safe code, so the host can act on it and log it without leaking tool
arguments or answer text.

## The five gates

| Hook | Blocks when |
|---|---|
| `before_tool_call` | policy denies it, evidence is missing, the tool is unknown to the manifest, or the arguments are invalid |
| `after_tool_result` | the claimed result is fabricated (vs the execution log) or implausible (vs the tool's declared return) |
| `before_final_answer` | the answer has no supporting evidence, or a canary tripped for it |
| `before_handoff` | the target agent is not permitted, or the payload is unsafe |
| `before_irreversible_action` | the action is irreversible and has no safeguard (no registered rollback, no human acknowledgement) → escalated |

```python
from director_ai.core.agent_preflight import AgentPreflightGuard, PreflightPolicy

guard = AgentPreflightGuard(
    PreflightPolicy(allowed_handoff_targets=frozenset({"billing", "support"}))
)

# Before a tool call
d = guard.before_tool_call("pay_invoice", {"invoice_id": "INV-1"}, manifest=manifest)
if d.blocked:
    refuse(d.reason)

# After the tool returns
d = guard.after_tool_result(
    "pay_invoice", {"invoice_id": "INV-1"}, claimed_result, execution_log=log
)

# Before the final answer
d = guard.before_final_answer(evidence_ok=bool(evidence), canary_tripped=tripped)

# Before handing off to another agent
d = guard.before_handoff("billing")

# Before an irreversible action
d = guard.before_irreversible_action(
    "permanently delete the account", rollback_registered=have_rollback
)
```

## Irreversible actions

`before_irreversible_action` scores the action's reversibility (via the
[`RuleReversibility`](#) estimator by default; any `ReversibilityEstimator` can
be injected). A reversible action is allowed. An irreversible one is allowed only
with a safeguard:

- a **registered rollback** — pair with the trajectory rollback manager: register
  a compensating action, then pass `rollback_registered=True` so the gate warns
  (armed) rather than escalating; or
- a **human acknowledgement** — `human_acknowledged=True`.

With neither, the action is **escalated** for a human to decide. The decision's
metadata carries the reversibility score that drove it.

## Through the guard

`ProductionGuard.preflight` returns a preflight guard whose result-plausibility
check uses the guard's coherence scorer:

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard.from_profile("finance")
decision = guard.preflight.before_tool_call(
    "pay_invoice", {"invoice_id": "INV-1"}, manifest=manifest
)
```

## Metrics

- `agent_preflight_decisions_total{hook, decision}` — every decision, labelled by
  hook point and outcome.
