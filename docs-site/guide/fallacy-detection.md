# Informal-fallacy detection

> **Status: core (Apache-2.0), heuristic, on by default in reasoning checks.** A
> match means "look here", not "this is fallacious" — the detector flags the
> surface markers of fallacy families and reports candidates for review.

The step-logic verifier catches *structural* faults — a conclusion that does not
follow, a circular step. This catches the named *informal* fallacies a fluent
answer slips in: attacking the person, appealing to the crowd, forcing a false
choice. It flags them from their characteristic phrasing.

## Families detected

| Family | Example marker |
|---|---|
| `ad_hominem` | "you're just biased", "he is incompetent" |
| `appeal_to_authority` | "because experts say so" |
| `bandwagon` | "everyone knows", "nobody believes" |
| `false_dichotomy` | "either with us or against", "only two options" |
| `hasty_generalization` | "this proves that all…" |
| `slippery_slope` | "will inevitably lead to", "next thing you know" |
| `appeal_to_emotion` | "think of the children", "you should be ashamed" |
| `post_hoc` | "rose after … therefore", "correlation … causes" |

Circular reasoning is deliberately omitted — the reasoning-chain verifier already
detects it by step overlap.

## Usage

```python
from director_ai.core import detect_fallacies, verify_reasoning_chain

detect_fallacies("Everyone knows this is best.").types     # -> ["bandwagon"]

result = verify_reasoning_chain("Step 1: … Step 2: everyone knows it works. …")
result.fallacies        # heuristic markers (FallacyMatch list)
```

Fallacies are reported on `result.fallacies` but, being lower-precision than the
structural checks, **do not** affect `result.chain_valid` — they are an advisory
signal alongside the high-precision step-logic and arithmetic verdicts. Disable
the pass with `verify_reasoning_chain(text, check_fallacies=False)`.

## Polyglot backend

The marker scan runs through the Rust `rust_detect_fallacies` kernel
(`backfire_core::compute::detect_fallacies`) with an identical pure-Python regex
pass. The patterns are deliberately lookaround- and backreference-free so the
Python `re` and Rust `regex` engines return the same matches in the same order —
verified bit-for-bit in the benchmark and tests. The dispatch is purely a speed
choice. See [Rust Acceleration](rust-acceleration.md).

## Measured

`python -m benchmarks.fallacy_detection`:

| Metric | Value |
|---|---|
| Detection accuracy (labelled markers + clean controls, n=20) | 1.00 |
| Control false-positive rate | 0.00 |
| Rust ↔ Python parity | exact |
| Rust speed-up over Python | ~5.6× |

These numbers are on a controlled marker set — real prose is noisier, and the
detector is a triage aid, not a proof. Genuine appeals to authority (a cited
study) read like the fallacious kind; a match means "review", not "wrong".
Numbers come from the committed benchmark and
`benchmarks/results/fallacy_detection.json`.
