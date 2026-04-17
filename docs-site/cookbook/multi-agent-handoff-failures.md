# Multi-Agent Handoff Failures

One agent's output is the next agent's prompt. When a researcher
hallucinates a number and hands off to a writer, the writer confidently
restates it; by the time a reviewer spots the error the final document
has already travelled three links down the chain. This cookbook shows
how to catch that failure at the handoff boundary with
`HandoffScorer` and how to halt the whole swarm with
`SwarmGuardian` when the same hallucination surfaces twice.

## Minimal reproduction

```python
from director_ai.agentic.swarm_guardian import SwarmGuardian
from director_ai.agentic.handoff_scorer import HandoffScorer
from director_ai.agentic.agent_profile import AgentProfile
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add(
    "company_revenue_2025",
    "ANULUM reported CHF 4.2M in revenue for FY2025 (audited).",
)
scorer = CoherenceScorer(ground_truth_store=store, threshold=0.6)

guardian = SwarmGuardian(scorer=scorer)
guardian.register(
    AgentProfile(name="researcher", threshold=0.65, allow_speculation=False),
)
guardian.register(
    AgentProfile(name="writer", threshold=0.55, allow_speculation=True),
)
handoff = HandoffScorer(scorer=scorer)

# Researcher hallucinates.
research_output = "ANULUM posted CHF 7.8M in 2025, up from CHF 3.1M in 2024."
verdict = handoff.score(
    sender="researcher",
    receiver="writer",
    message=research_output,
    context="What was ANULUM's 2025 revenue?",
)

if verdict.halted:
    print(f"BLOCKED at handoff — {verdict.reason}")
    # Writer never sees the wrong number; swarm is notified.
    guardian.report_halt(agent="researcher", evidence=verdict)
else:
    writer_output = writer_agent(research_output)
    # score writer's output too — handoff chains every link
```

## What this catches

* **Numeric drift**: `7.8M` vs KB's `4.2M` trips the numeric-consistency
  rule in the rules backend and the NLI contradiction signal at the
  same time.
* **Fabricated comparisons**: `"up from 3.1M"` has no source document —
  `HandoffScorer` returns `halted=True` with `reason="no_grounding"`.
* **Identity reuse across turns**: if the researcher repeats the
  halucinated number after one halt, `SwarmGuardian.report_halt`
  triggers a cascade halt on every downstream agent in the swarm.

## Diagnosing with the swarm metrics

```python
from director_ai.agentic.swarm_metrics import SwarmMetrics

metrics = SwarmMetrics()
metrics.record(agent="researcher", verdict=verdict)
metrics.record(agent="writer", verdict=next_verdict)

report = metrics.snapshot()
print(report.per_agent["researcher"].halt_rate)      # 0.42
print(report.handoff_failure_rate)                    # 0.33
print(report.cascade_halts)                           # 1
```

A rising `halt_rate` for one agent while others stay flat is a sign
that agent needs either a tighter `AgentProfile.threshold` or a
broader knowledge base. Cascade halts > 0 are an escalation signal:
the same hallucination survived one handoff and was caught later.

## Framework adapters

`director_ai.integrations` ships adapters that plug the handoff scorer
into LangGraph, CrewAI, OpenAI Swarm, and AutoGen without changing
their agent APIs. The adapter wraps every edge/hand-off callback and
raises a framework-native exception on halt.

```python
from director_ai.integrations.langgraph import DirectorLangGraphGuard

graph = langgraph.Graph()
graph.add_edge("researcher", "writer", guard=DirectorLangGraphGuard(handoff))
```

Repository locations:

* `src/director_ai/agentic/{swarm_guardian,handoff_scorer,agent_profile,swarm_metrics}.py`
* Tests: `tests/test_swarm_guardian.py`, `tests/test_handoff_scorer.py`,
  `tests/test_agent_profile.py`, `tests/test_swarm_metrics.py`

## Operating notes

* Run the swarm guard in-process with the scorer; cross-process RPC
  doubles tail latency on every handoff.
* `HandoffScorer` defaults to `threshold = min(sender.threshold,
  receiver.threshold)`. Override per edge when the receiver has a
  higher confidence requirement (e.g. legal writer vs draft writer).
* `SwarmGuardian.on_halt` accepts a callback; wire it to your incident
  channel so a cascade halt pages a human.
* If the same agent trips halts every run, the fix is almost always
  to add missing facts to the knowledge base — not to lower the
  threshold.
