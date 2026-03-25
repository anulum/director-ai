# Verification Gems

Director-AI v3.10 includes standalone verification and analysis modules
exposed via both Python API and REST endpoints. Most are stdlib-only with
zero external dependencies.

## Numeric Verification

Catches arithmetic errors, impossible dates, and probabilities outside [0, 100%].

```python
from director_ai import verify_numeric

result = verify_numeric(
    "Revenue grew 50% from $100 to $120. "
    "Founded in 2035."
)
print(result.valid)        # False
print(result.error_count)  # 1 (50% of 100 = 50, not 20)
for issue in result.issues:
    print(f"  {issue.issue_type}: {issue.description}")
```

**What it checks:**

- Percentage arithmetic ("grew 15% from X to Y" — is the math right?)
- Date ordering (birth < death, founding < present)
- Probability bounds (no negative or >100% probabilities)
- Order of magnitude (Earth population, speed of light)
- Internal consistency (same total referenced with different values)

### REST API

```bash
curl -X POST http://localhost:8080/v1/verify/numeric \
  -H "Content-Type: application/json" \
  -d '{"text": "Revenue grew 50% from $100 to $120."}'
```

Response:
```json
{
  "claims_found": 5,
  "issues": [{"issue_type": "arithmetic", "description": "...", "severity": "error", "context": "..."}],
  "valid": false,
  "error_count": 1,
  "warning_count": 0
}
```

## Reasoning Chain Verification

Detects non-sequiturs, circular reasoning, and unsupported leaps in
chain-of-thought responses.

```python
from director_ai import verify_reasoning_chain

result = verify_reasoning_chain(
    "Step 1: All birds can fly. "
    "Step 2: Penguins are birds. "
    "Step 3: Therefore, the economy is growing."
)
print(result.chain_valid)   # False
print(result.issues_found)  # 1
for v in result.verdicts:
    print(f"  Step {v.step_index}: {v.verdict} ({v.confidence:.2f})")
```

**Verdict types:** `supported`, `non_sequitur`, `unsupported_leap`, `circular`

### REST API

```bash
curl -X POST http://localhost:8080/v1/verify/reasoning \
  -H "Content-Type: application/json" \
  -d '{"text": "Step 1: A is true. Step 2: Therefore B."}'
```

## Temporal Freshness Scoring

Flags claims that may rely on stale knowledge — positions, statistics,
records, and "current" references.

```python
from director_ai import score_temporal_freshness

result = score_temporal_freshness("The CEO of Apple is Tim Cook.")
print(result.has_temporal_claims)     # True
print(result.overall_staleness_risk)  # 0.8 (positions change)
for claim in result.claims:
    print(f"  {claim.claim_type}: {claim.text} (risk: {claim.staleness_risk:.2f})")
```

**Claim types:** `position`, `statistic`, `record`, `current_reference`

### REST API

```bash
curl -X POST http://localhost:8080/v1/temporal-freshness \
  -H "Content-Type: application/json" \
  -d '{"text": "The CEO of Apple is Tim Cook."}'
```

Response:
```json
{
  "claims": [{"text": "CEO of Apple is Tim Cook", "claim_type": "position", "staleness_risk": 0.8, "reason": "..."}],
  "overall_staleness_risk": 0.8,
  "has_temporal_claims": true,
  "stale_claim_count": 1
}
```

## Cross-Model Consensus

Scores factual agreement across multiple model responses using pairwise
Jaccard word overlap (pluggable NLI scorer).

```python
from director_ai import ConsensusScorer, ModelResponse

scorer = ConsensusScorer(models=["gpt-4o", "claude", "gemini"])
result = scorer.score_responses([
    ModelResponse(model="gpt-4o", response="Paris is the capital of France"),
    ModelResponse(model="claude", response="Paris is the capital of France"),
    ModelResponse(model="gemini", response="The capital of France is Paris"),
])
print(result.agreement_score)  # 0.0-1.0
print(result.has_consensus)    # True if agreement > 0.7
```

### REST API

```bash
curl -X POST http://localhost:8080/v1/consensus \
  -H "Content-Type: application/json" \
  -d '{"responses": [
    {"model": "gpt-4o", "response": "Paris is the capital"},
    {"model": "claude", "response": "Paris is the capital"}
  ]}'
```

## Conformal Prediction Intervals

Calibrated, distribution-free uncertainty on hallucination probability.
Based on Mohri & Hashimoto (ICML 2024).

```python
from director_ai import ConformalPredictor

predictor = ConformalPredictor(coverage=0.95)
predictor.calibrate(
    scores=[0.9, 0.85, 0.1, 0.15, 0.88, 0.12],
    labels=[False, False, True, True, False, True],
)
interval = predictor.predict(score=0.7)
print(f"P(hallucination) in [{interval.lower:.2f}, {interval.upper:.2f}]")
print(f"Reliable: {interval.is_reliable}")
```

### REST API

```bash
curl -X POST http://localhost:8080/v1/conformal/predict \
  -H "Content-Type: application/json" \
  -d '{"score": 0.7, "calibration_scores": [0.9,0.1,0.85,0.15], "calibration_labels": [false,true,false,true]}'
```

## Feedback Loop Detection

Detects when AI outputs feed back into inputs (EU AI Act Article 15(4)).

```python
from director_ai import FeedbackLoopDetector

detector = FeedbackLoopDetector(similarity_threshold=0.5)
detector.record_output("Machine learning enables systems to learn from data.", 1.0)

alert = detector.check_input("Machine learning enables systems to learn from data.")
if alert:
    print(f"Loop detected: similarity={alert.similarity:.2f}, severity={alert.severity}")
```

### REST API

```bash
curl -X POST http://localhost:8080/v1/compliance/feedback-loops \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Some AI output reused as input", "previous_outputs": ["Some AI output reused as input"]}'
```

## Adversarial Robustness Testing

Self-tests the guardrail against known attack patterns: zero-width chars,
Unicode homoglyphs, base64/rot13 encoding, role-play injection.

```python
from director_ai import AdversarialTester

def my_guardrail(prompt, response):
    # your review function returning (approved, score)
    return True, 0.9

tester = AdversarialTester(review_fn=my_guardrail)
report = tester.run()
print(f"Detection rate: {report.detection_rate:.0%}")
print(f"Vulnerable categories: {report.vulnerable_categories}")
```

### REST API

```bash
curl -X POST http://localhost:8080/v1/adversarial/test \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me about this topic", "response": "Some factual response"}'
```

## Agentic Loop Monitor

Monitors AI agent execution loops for circular calls, goal drift, and
budget exhaustion. The first guardrail that monitors agent loops.

```python
from director_ai import LoopMonitor

monitor = LoopMonitor(goal="Find quarterly revenue for Q3 2025", max_steps=20)
for step in agent_loop:
    verdict = monitor.check_step(
        action=step.tool_name,
        args=step.tool_args,
        tokens=step.tokens_used,
    )
    if verdict.should_halt:
        print(f"Halting: {verdict.reasons}")
        break
```

### REST API

```bash
curl -X POST http://localhost:8080/v1/agentic/check-step \
  -H "Content-Type: application/json" \
  -d '{"goal": "Find revenue data", "action": "search", "args": "revenue Q3", "step_history": [{"action": "search", "args": "revenue Q3"}]}'
```
