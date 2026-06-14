# Chain-of-thought arithmetic verification

> **Status: core (Apache-2.0), deterministic, on by default in reasoning checks.**
> Every recognised equation is evaluated and compared to its asserted result — no
> model, no heuristic, no tolerance games beyond floating-point rounding.

A model can reason fluently and still get the sums wrong. Where
[numeric verification](scoring.md) checks numeric *plausibility* (percentage
changes, date logic, magnitudes), this checks whether the explicit arithmetic in
a chain-of-thought answer actually holds: `3 + 4 = 8` is flagged, `12 × 5 = 60`
passes.

## How it works

1. Equations are extracted with a regex — `left = right`, `left equals right`,
   `left is equal to right` — where the left side must carry at least one
   operator, so a plain `x is 5` is not mistaken for arithmetic.
2. The left side is evaluated and compared to the asserted right side within a
   floating-point tolerance (`math.isclose`).
3. Thousands separators (`1,000`), currency (`$3,000`), parentheses, unary
   minus, and the Unicode operators `× ÷ ·` are all understood; scientific
   notation and anything outside the arithmetic character set make an expression
   unevaluable (and so simply uncounted, never falsely flagged).

It is wired into the reasoning-chain verifier, so step-logic and arithmetic are
checked together:

```python
from director_ai.core import verify_arithmetic, verify_reasoning_chain

verify_arithmetic("Step 2: 3 + 4 = 8.").errors          # -> [3 + 4 = 8]
result = verify_reasoning_chain("Step 1: ... Step 2: 3 + 4 = 8. ...")
result.math_errors      # arithmetic mistakes
result.chain_valid      # False — a wrong sum is a chain issue
```

Disable the arithmetic pass with `verify_reasoning_chain(text, check_math=False)`.

## Polyglot backend

The evaluator runs through the Rust `rust_eval_arithmetic` kernel
(`backfire_core::compute::eval_arithmetic`) with a pure-Python `ast` fallback.
The two are **bit-for-bit identical**: the same IEEE-754 doubles, the same
precedence and left-associativity, the same character set, and any non-finite
result (division by zero, overflow) reported as `NaN` on both paths. The dispatch
is therefore purely a speed choice. See [Rust Acceleration](rust-acceleration.md).

## Measured

`python -m benchmarks.math_consistency`:

| Metric | Value |
|---|---|
| Detection accuracy (labelled CoT snippets, n=12) | 1.00 |
| Error precision / recall | 1.00 / 1.00 |
| Rust ↔ Python parity | exact (NaN included) |
| Rust speed-up over Python | ~16× |

Detection is exact because the check is deterministic arithmetic, not a learned
judgement — the only failure mode is an equation the extractor does not
recognise, which is counted as "not checked", never as "correct". Numbers come
from the committed benchmark and `benchmarks/results/math_consistency.json`.
