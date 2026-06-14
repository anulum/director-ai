# Cross-model consensus

> **Status: panel-level agreement scoring.** The consensus engine compares
> several models' answers to one prompt, quantifies how much they agree, and
> names the claims where they diverge. It is a routing and review aid — it tells
> you when one model's answer should not be trusted alone — while the streaming
> halt and NLI scorer remain the response-side ground truth.

Every other guard in Director-AI scores a *single* response. `CrossModelConsensus`
scores a *panel*: give it the same question answered by GPT, Claude, Gemini, a
local Llama, or any other set of models and it returns how much the answers agree,
the specific claims where they conflict (with evidence), and whether to accept the
majority answer, send it for review, or escalate to a stronger model or a human.

## The two agreement signals

| Signal | When it is used | What it measures |
|---|---|---|
| **semantic agreement** | an NLI contradiction scorer is supplied | `1 - max(P(contradiction) a→b, P(contradiction) b→a)` for each answer pair — the directional maximum, because either direction is enough to break agreement |
| **lexical agreement** | no scorer supplied (or as a cheap pre-filter) | Jaccard word overlap of the two answers via the Rust `rust_word_overlap` kernel, with a bit-exact Python fallback |

The panel **consensus** is the mean pairwise agreement. Below `escalate_threshold`
(default 0.45) the recommendation is **escalate**; below `accept_threshold`
(default 0.7) it is **review**; at or above it is **accept**. The result also
carries the full pairwise agreement matrix and a short rationale.

## Divergence with evidence

When an NLI scorer is supplied, divergences are reported at the **claim** level:
each answer is split into claims and every cross-model claim pair whose
contradiction probability clears the flag threshold (the scorer's own `threshold`
by default) is returned as a `Divergence` naming the two models, the two
conflicting claims, and the contradiction strength — so "the models disagree"
always comes with the sentence that caused it. Without a scorer the engine still
surfaces the single least-agreeing answer pair so the caller sees who diverged
most.

## Usage

```python
from director_ai.guard import ProductionGuard
from director_ai.core.consensus import ModelResponse

guard = ProductionGuard()
panel = [
    ModelResponse("gpt", "The treaty was signed in 1920."),
    ModelResponse("claude", "The treaty was signed in 1919."),
    ModelResponse("gemini", "The treaty was signed in 1920."),
]

# Lexical-only (dependency-free, offline):
result = guard.cross_model_consensus(panel)
print(result.consensus, result.recommendation, result.rationale)

# Semantic, claim-level divergence with evidence:
from director_ai.core.scoring.contradiction import ContradictionScorer

nli = ContradictionScorer.from_pretrained()
result = guard.cross_model_consensus(panel, nli=nli)
for d in result.divergences:
    print(f"{d.model_a} vs {d.model_b}: {d.contradiction:.2f}")
    print(f"  {d.claim_a!r}  ⟂  {d.claim_b!r}")
```

The engine (and any supplied scorer) persists for the life of the guard.

## Polyglot backend

The lexical agreement runs through the Rust `backfire_kernel.rust_word_overlap`
extension when it is installed, and a pure-Python Jaccard fallback otherwise. The
two are **bit-for-bit identical** — both case-fold, split on whitespace, and
retain punctuation on the token — so the dispatch is purely a speed choice and the
engine runs everywhere. See [Rust Acceleration](rust-acceleration.md).

## Measured behaviour

On the bundled labelled panels (`python -m benchmarks.cross_model_consensus`):

| Metric | Value |
|---|---|
| Lexical recommendation accuracy (accept vs escalate panels) | 1.00 |
| Rust ↔ Python parity | exact |
| Throughput (lexical consensus) | ~114,000 panels/s |

These numbers come from the committed benchmark and
`benchmarks/results/cross_model_consensus.json`; reproduce them with the command
above. The benchmark exercises the offline lexical path; the semantic NLI path is
validated by the unit tests with a deterministic scorer.
