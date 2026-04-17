# Director-AI Formal Models

Machine-checked Lean 4 models of safety-critical components in
Director-AI. The goal is not to reimplement production logic in Lean,
but to isolate the few pieces whose behaviour is a safety claim — and
prove the claim once, against a minimal model, so that any future
refactor has to justify why the Python behaviour still matches.

## Layout

```
formal/
├── HaltMonitor/                      # Lake project
│   ├── lakefile.toml
│   ├── lean-toolchain                # pins Lean 4.29.1
│   ├── HaltMonitor.lean              # entry — imports Core + Properties
│   └── HaltMonitor/
│       ├── Core.lean                 # model of the halt loop
│       └── Properties.lean           # safety theorems
└── README.md                         # this file
```

## HaltMonitor

The Python loop in
`src/director_ai/core/runtime/kernel.py::HaltMonitor.stream_output`
is the safety floor of Director-AI. It decides, per token, whether
the current coherence score still clears `hard_limit`. Everything
else in the file — thread event, timeouts, callbacks — is
orchestration; a bug there degrades reliability but does not let a
below-threshold token through. The threshold check does, so that is
the only piece we model.

### What is proved

Let `run : Score → List (Token × Score) → Output` walk an input
stream, comparing each score to `hardLimit`, and return either
`emitted tokens` (full stream passed) or `halted prefix` (some
token failed the check).

1. **`run_emitted_preserves_input`** — when `run ...` returns
   `emitted ts`, `ts` is exactly the token projection of the input.
   No token is fabricated, duplicated, or reordered.
2. **`run_emitted_implies_all_pass`** — when `run ...` returns
   `emitted ts`, every input item has a score `≥ hardLimit`.
3. **`run_any_fail_implies_not_emitted`** — if any input item has a
   score `< hardLimit`, no emitted output exists.
4. **`run_any_fail_implies_halted`** — strengthened: the result is
   `halted prefix` for some prefix.

Together these formalise "no token whose score is below
`hard_limit` is ever emitted" — the safety claim made in the
HaltMonitor docstring.

### What is NOT proved

- Token and total timeouts (orchestration, not safety).
- Thread-safety of `_active.set/clear` (orchestration).
- Behaviour under a faulty `coherence_callback` that raises
  exceptions (the Python implementation halts; the model assumes
  the callback returns).
- Floating-point precision. The model uses `ℚ` so comparisons are
  exact; Python uses `float`, which can round. The precision
  boundary is a Python-side concern — a scorer returning
  `hard_limit - ε` for tiny `ε` is already well below the design
  tolerance.

### Building and checking

```bash
cd formal/HaltMonitor
lake build       # compiles and checks every proof
```

No network access required after the Lean toolchain is installed
via `elan`. Typical check time is under ten seconds on the mining
rig. A failure means a proof broke, not a flaky run — the
theorems are machine-checked with no nondeterminism.

## Adding another model

1. Create `formal/<Component>/` with its own `lakefile.toml` and
   `lean-toolchain`.
2. Build a minimal `Core` that captures the safety-relevant
   behaviour only. Resist the urge to recreate the full Python
   class — proofs get harder and the model drifts from the thing
   you actually care about.
3. State the safety claim as a theorem in `Properties`, prove it,
   and cite the Python function it models.
4. Reference the new project from this README.

The bar for adding a model is "there is a claim we want to hold
across refactors that cannot be expressed as a unit test". Type
signatures, guard conditions, and fallback logic tend to qualify;
performance or integration behaviour does not.
