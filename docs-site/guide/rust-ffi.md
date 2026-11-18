<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-AI — Rust Python bindings and scoring boundaries
-->

# Rust Python Bindings

`backfire-kernel` exposes Rust scoring, streaming and numerical operations as
the Python module `backfire_kernel` through PyO3. The Python bindings accept
owned token strings and invoke Python callbacks synchronously; this interface
does not provide a zero-copy token or model-inference guarantee.

## Build from source

Use a Python virtual environment satisfying the binding package's Python
requirement and the Rust workspace toolchain. The binding's
`backfire-kernel/crates/backfire-ffi/pyproject.toml` declares Python 3.11 or
newer and Maturin `>=1.15,<2.0`.

From the repository root, with a compatible Maturin installed in that
environment:

```bash
maturin develop --release --locked \
  --manifest-path backfire-kernel/crates/backfire-ffi/Cargo.toml
python -c "import backfire_kernel; print(backfire_kernel.__file__)"
```

The distribution name is `backfire-kernel`; the import name is
`backfire_kernel`. Check the imported module's path when multiple
environments or locally built extensions exist. A successful import alone
does not establish that a binary was built from the current source.

## Streaming kernels

```python
from backfire_kernel import BackfireConfig, RustSafetyKernel, RustStreamingKernel

basic = RustSafetyKernel(hard_limit=0.5)
output = basic.stream_output(["Hello ", "world."], lambda token: 0.9)

streaming = RustStreamingKernel(
    config=BackfireConfig(
        hard_limit=0.5,
        window_size=10,
        window_threshold=0.55,
    ),
)
session = streaming.stream_tokens(["The ", "sky ", "is ", "blue."], lambda token: 0.9)
print(session.halted, session.avg_coherence())
```

The constant callbacks above demonstrate token gating only. They do not
verify any claim. These direct native callbacks receive the current token.
The Python `StreamingKernel` orchestrator scores accumulated response text
and provides additional callback, evidence and async behaviour; see
[Streaming halt](streaming.md). A halt does not undo already delivered text
or an external action.

## Scoring and grounding

The high-level `CoherenceScorer` accepts `scorer_backend="rust"` and its
alias `"backfire"`. Both use the actual Rust scorer. The current wrapper
supplies a retrieval callback but no model-backed NLI callback, so their
logical and factual signals use native heuristics. They are not calibrated
NLI contradiction probabilities.

```python
from director_ai import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("capital", "Paris is the capital of France.", tenant_id="example")
scorer = CoherenceScorer(scorer_backend="rust", use_nli=False, ground_truth_store=store)
approved, score = scorer.review(
    "capital", "Paris is the capital of France.", tenant_id="example"
)
```

The wrapper binds `tenant_id` to the synchronous retrieval callback for each
native call, including calls reached through `review_batch()`, `areview()`
and `calculate_factual_divergence()`. The previous context is restored when
the call returns or raises; concurrent execution contexts remain separate.
An empty tenant preserves the store's unscoped retrieval API. The caller
must authenticate and authorise tenant identity before passing it here.

With no configured store, the wrapper explicitly returns missing context to
Rust; native demo facts are not used as application grounding. Missing
context yields the native neutral factual divergence of `0.5`.

Two model policies matter:

- `require_model_backed_nli=True` raises `RuntimeError` from `review()`,
  `review_batch()` and `areview()` when the configured path lacks a model.
  The current `rust` and `backfire` wrapper paths cannot satisfy this policy.
- `strict_mode=True` uses the shared missing-model rejection path rather
  than accepting native heuristic results. A review is rejected and marked
  `strict_mode_rejected` and `degraded_mode`.

For model-backed factual checks, configure a supported NLI backend and
require it explicitly; see the [scorer API](../api/scorer.md).

### Direct native scorer

The lower-level `RustCoherenceScorer` constructor accepts `config`,
`nli_callback` and `knowledge_callback`. Omitting `nli_callback` selects
`HeuristicNli`; omitting `knowledge_callback` selects the native in-memory
store, which contains demo facts. Its `review(prompt, action)` method has no
tenant argument. A direct caller must supply and scope its own real
retrieval callback. Providing a callback does not by itself establish model
quality, tenant authorisation or an audit trail.

## Callback boundaries

The current bindings enter Python with `Python::attach` for each callback.
Conversion failures and Python exceptions have different defaults:

| Callback | Binding result after an exception or invalid return type |
|----------|---------------------------------------------------------|
| Kernel coherence callback | `0.0` |
| Native scorer NLI callback | `0.5` |
| Native scorer knowledge callback | Missing context (`None`) |

These defaults do not prove a contradiction or model availability.
`BackfireConfig` validates values before storing them, and the kernels
normalise non-finite callback scores to `0.0`. Timing checks are cooperative:
a synchronous callback must return before the caller can inspect elapsed
time. A configured time budget cannot pre-empt a blocked callback.

## Other exposed operations

The extension also exposes phase dynamics, geometry and observer interfaces,
including `UPDEStepper`, `L16Controller`, `SSGFEngine`, `TCBOObserver` and
`PGBOEngine`. Their presence in the extension does not qualify them as part
of the factual-scoring path.

## Performance evidence

The repository's `backfire-kernel/BENCHMARKS.md` contains a dated
2026-02-22 Windows/Rust benchmark report. Its measurements describe that
recorded environment and source, not the current Python callback boundary
or a deployment latency guarantee. Run the applicable native and
cross-language benchmarks against the source and environment being
deployed before making a performance comparison.

The browser/Worker package is a separate interface; see
[WASM runtime](../deployment/wasm-runtime.md).
