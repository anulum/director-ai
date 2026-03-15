# Migration Guide: v2 → v3

This guide covers every breaking change between Director-AI v2.x and v3.x with before/after code.

## 1. Enterprise Imports (Lazy-Loading)

Enterprise modules now load lazily to avoid pulling heavy dependencies (`redis`, `pyjwt`, `argon2-cffi`, `psycopg2`) on import.

=== "v2 (removed)"

    ```python
    from director_ai.core import TenantRouter, PolicyEngine, AuditLogger
    ```

=== "v3"

    ```python
    from director_ai.enterprise import TenantRouter, PolicyEngine, AuditLogger
    ```

The `director_ai.enterprise` package loads only when accessed. If enterprise extras aren't installed, you get a clear `ImportError` with install instructions instead of a cryptic missing-module traceback.

## 2. CoherenceScorer Validation

`CoherenceScorer` now raises `ValueError` on invalid threshold combinations instead of silently clamping.

=== "v2 (silent clamp)"

    ```python
    scorer = CoherenceScorer(threshold=0.9, soft_limit=0.3)
    # v2: silently set soft_limit = threshold (0.9)
    ```

=== "v3 (explicit error)"

    ```python
    scorer = CoherenceScorer(threshold=0.9, soft_limit=0.3)
    # v3: ValueError("soft_limit (0.3) must be >= threshold (0.9)")
    ```

**Fix:** ensure `soft_limit >= threshold` in all call sites.

## 3. StreamingKernel Constructor

Two parameters renamed for clarity.

| v2 Parameter | v3 Parameter | Reason |
|-------------|-------------|--------|
| `check_interval` | `score_every_n` | Clarifies unit (tokens, not seconds) |
| `halt_on_fail` | `halt_on_breach` | Aligns with threshold terminology |

=== "v2"

    ```python
    kernel = StreamingKernel(
        scorer=scorer,
        check_interval=5,
        halt_on_fail=True,
    )
    ```

=== "v3"

    ```python
    kernel = StreamingKernel(
        scorer=scorer,
        score_every_n=5,
        halt_on_breach=True,
    )
    ```

## 4. Score Object Fields

`ScoreResult.contradiction_score` renamed to `ScoreResult.h_logical` and `ScoreResult.retrieval_score` renamed to `ScoreResult.h_factual`.

=== "v2"

    ```python
    result = scorer.review(query, response)
    print(result.contradiction_score)
    print(result.retrieval_score)
    ```

=== "v3"

    ```python
    result = scorer.review(query, response)
    print(result.h_logical)
    print(result.h_factual)
    ```

## 5. Configuration Profiles

v3 replaces scattered kwargs with `DirectorConfig` profiles.

=== "v2 (kwargs everywhere)"

    ```python
    scorer = CoherenceScorer(
        threshold=0.6,
        soft_limit=0.7,
        use_nli=True,
        nli_model="yaxili96/FactCG-DeBERTa-v3-Large",
        nli_device="cuda",
        cache_size=2048,
    )
    ```

=== "v3 (config object)"

    ```python
    from director_ai import DirectorConfig, CoherenceScorer

    config = DirectorConfig(
        threshold=0.6,
        soft_limit=0.7,
        scorer_backend="deberta",
        nli_device="cuda",
        cache_size=2048,
    )
    scorer = CoherenceScorer(config=config)
    ```

Both styles work in v3 — kwargs are still accepted. But `DirectorConfig` enables serialization (YAML/JSON), validation, and reuse across scorer/kernel/agent.

## 6. New Features in v3

No migration needed — these are additive.

| Feature | Module | Guide |
|---------|--------|-------|
| `review_batch()` | `CoherenceScorer` | [Benchmarks — Batch Coalescing](../benchmarks.md#batch-coalescing) |
| ONNX export | `director_ai.core.backends.OnnxBackend` | [ONNX Export](onnx-export.md) |
| Rust FFI kernel | `backfire-kernel` crate | [Rust FFI](rust-ffi.md) |
| Hybrid LLM judge | `CoherenceAgent(judge=...)` | [Scoring](scoring.md) |
| Config profiles | `DirectorConfig` | [Configuration](config.md) |
| Bidirectional NLI | `CoherenceScorer(bidirectional=True)` | [Threshold Tuning](threshold-tuning.md) |
| Local DeBERTa judge | `CoherenceAgent(judge="local")` | [Benchmarks](../benchmarks.md) |

## Quick Checklist

- [ ] Update enterprise imports to `from director_ai.enterprise import ...`
- [ ] Ensure `soft_limit >= threshold` in all `CoherenceScorer` calls
- [ ] Rename `check_interval` → `score_every_n` in `StreamingKernel`
- [ ] Rename `halt_on_fail` → `halt_on_breach` in `StreamingKernel`
- [ ] Rename `contradiction_score` → `h_logical`, `retrieval_score` → `h_factual`
- [ ] (Optional) Migrate to `DirectorConfig` for centralized configuration

---

*Questions? Open an [issue](https://github.com/anulum/director-ai/issues/new) or check [Troubleshooting](troubleshooting.md).*
