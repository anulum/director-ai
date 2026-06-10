# Tier-6 Reasoning Escalation

`ReasoningScorer` is an escalation-only reasoning tier above the NLI scorer
(Tier 5). The NLI scorer is fast but verdict-only — a single divergence number
with no rationale and no harm taxonomy. This tier adds a causal-LM safety
chain-of-thought that fires **only** when the lower tier is borderline, so the
median request never pays for it.

## When it fires

The tier is consulted only when the composite coherence score sits within
`escalation_margin` (default 0.15) of the decision boundary — the band where the
lower tier's approve/reject call is least certain:

```text
        reject            borderline            approve
   ───────────────|===================|───────────────►  coherence
                threshold-m         threshold+m
                  └── reasoning tier fires here ──┘
```

Confident scores on either side skip the tier entirely. It is **disabled by
default**; enabling it is purely additive.

## Structured verdict

When it fires, the reasoning model returns a `ReasoningVerdict`:

| Field | Meaning |
| --- | --- |
| `approved` | the reasoning approve/reject decision |
| `confidence` | 0–1 confidence in the verdict |
| `rationale` | one-sentence explanation |
| `harm_category` | a canonical [`HarmCategory`](sanitizer.md) (HarmBench taxonomy) or `None` |
| `detected_issues` | short specific issue strings |
| `adjusted_score` | the blended composite coherence |

The model's free-form category string is normalised through the same
`to_harm_category()` taxonomy the input sanitizer uses, so a Tier-6 verdict and a
sanitizer block speak the same seven-category language.

The verdict blends into the lower-tier score at the same 30/70 confidence-scaled
ratio the LLM judge uses. Approval then requires **both** the blended score to
clear the threshold **and** the reasoning verdict to approve — so a confident
safety rejection halts a borderline output, while an unavailable backend or an
unparsable reply leaves the lower-tier verdict untouched (the tier never silently
flips a decision on failure).

## Backends

Three backends mirror the [LLM judge](guard.md):

- `"local"` — a local causal-LM loaded with `transformers` (no API calls);
- `"openai"` — OpenAI Chat Completions;
- `"anthropic"` — Anthropic Messages.

With an external provider, the borderline prompt/response is sent to that
provider; set `privacy_mode=True` to redact PII first.

## Enabling

Via `DirectorConfig`:

```python
from director_ai.core.config import DirectorConfig

config = DirectorConfig(
    reasoning_enabled=True,
    reasoning_provider="local",          # or "openai" / "anthropic"
    reasoning_model="Qwen/Qwen2.5-7B-Instruct",
    reasoning_escalation_margin=0.15,
)
scorer = config.build_scorer()
```

Or directly:

```python
from director_ai.core import CoherenceScorer

scorer = CoherenceScorer(
    threshold=0.5,
    reasoning_enabled=True,
    reasoning_provider="anthropic",
)
approved, score = scorer.review(prompt, response)
if score.reasoning_escalated:
    print(score.reasoning_harm_category, score.reasoning_rationale)
```

## Full API

::: director_ai.core.scoring.reasoning_scorer.ReasoningScorer

::: director_ai.core.scoring.reasoning_scorer.ReasoningVerdict
