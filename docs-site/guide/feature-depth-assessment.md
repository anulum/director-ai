# Feature Depth Assessment

Honest audit of two features that external reviewers flagged as potentially
"scaffolding" or "marketing claims": the agent loop monitor and online
calibration system. Both were audited at source level on 2026-04-05.

---

## 1. Agent Loop Monitor

**Location:** `src/director_ai/agentic/loop_monitor.py` (240 lines)

### What It Does

Five independent detection mechanisms running per agent step:

| Mechanism | Implementation | Trigger |
|-----------|---------------|---------|
| Circular tool calls | Counter-based action hash tracking | Same tool+args repeated ≥3× |
| Goal drift | Jaccard similarity between goal words and action | Similarity drops below threshold |
| Budget exhaustion | Cumulative token tracking | Warns at 90%, halts at 100% |
| Wall-clock timeout | Monotonic time comparison | Elapsed > max_seconds |
| Step count limit | Simple counter | Steps > max_steps (default 50) |

### Maturity

- **Tests:** 18 test classes, 181 lines (circular detection, drift scoring, budget, timeout, step limits, loop status)
- **Integration:** Wired into `server.py` as HTTP endpoint `/v1/agent/step` — receives step history, returns `StepVerdict` with halt/warn/continue
- **CLI integration:** Used in `_cli_verify.py` for goal-based verification
- **Verdict:** **Production-ready.** Not scaffolding.

### Known Limitations

- Default goal drift scorer uses Jaccard word overlap (fast, simple). For semantic drift detection, the scorer is pluggable — replace with NLI-based callback for production. Documented in docstring.
- No persistent state across requests — each `/v1/agent/step` call receives full step history. Stateless by design (the caller tracks state).

---

## 2. Online Calibration (FeedbackStore + OnlineCalibrator)

**Location:**
- `src/director_ai/core/calibration/feedback_store.py` (188 lines)
- `src/director_ai/core/calibration/online_calibrator.py` (164 lines)

### What It Does

**FeedbackStore** — SQLite3 database (WAL mode, thread-safe) storing human feedback:
- `report()` — stores (prompt, response, guardrail_score, guardrail_approved, human_approved, domain, timestamp)
- `get_corrections()` — retrieves with domain filter
- `get_disagreements()` — filters where guardrail ≠ human (FP/FN)
- `export_training_data()` — outputs fine-tuning format

**OnlineCalibrator** — threshold optimisation from feedback:
- Computes confusion matrix (TP, TN, FP, FN) from guardrail vs. human verdicts
- Calculates TPR, TNR, FPR, FNR with **Wilson 95% confidence intervals**
- **Threshold sweep:** tests 91 thresholds (5%–95%), finds threshold maximising balanced accuracy `(TPR + TNR) / 2`
- Returns `CalibrationReport` with `optimal_threshold` (requires ≥20 corrections with scores)

### Maturity

- **Tests:** 24 test classes, 343 lines (empty store, perfect guardrail, all-FP, mixed errors, threshold separation, domain filtering, performance bounds)
- **Integration:** Wired into `ProductionGuard` class in `guard.py`, paired with `ConformalPredictor` for confidence intervals
- **Demo:** `examples/online_calibration_demo.py` — realistic 25-correction scenario with per-domain calibration
- **Verdict:** **Production-ready.** Real threshold optimisation, real database, real statistics.

### Known Limitations

- `FeedbackStore.report()` must be called explicitly by the integration layer — it does not auto-hook into the scoring pipeline. The caller decides when to record feedback.
- Calibrated threshold is computed on-demand via `calibrate()` — it is not persisted or auto-applied. The caller must read the `CalibrationReport.optimal_threshold` and update their scorer configuration.
- Minimum 20 corrections required for threshold recommendation. Below that, returns `None`.

---

## Side-by-Side Summary

| Aspect | Loop Monitor | Online Calibration |
|--------|--------------|-------------------|
| Implementation | 240 lines | 352 lines |
| Tests | 18 classes (181 lines) | 24 classes (343 lines) |
| Scaffolding? | No — fully functional | No — fully functional |
| Server integration | `/v1/agent/step` endpoint | `ProductionGuard` class |
| Statistical rigour | Jaccard heuristic (pluggable) | Wilson CIs, balanced accuracy |
| Known gaps | Default scorer is simple | Manual report() wiring |

---

## Conclusion

Both features are **implemented, tested, and integrated** — not stubs or future
placeholders. The implementation is intentionally pragmatic: simple defaults
(Jaccard for drift, SQLite for feedback) with escape hatches for heavier
backends (NLI scorer, external databases). This is a deliberate design choice,
not incomplete work.

External reviewers questioning depth should inspect the test suites:
- `tests/test_loop_monitor.py`
- `tests/test_calibration.py`
