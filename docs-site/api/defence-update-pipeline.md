# Defence Update Pipeline

`DefenseUpdatePipeline` is the reviewed promotion gate between three
experimental hardening surfaces:

- `SelfImprovingGuardLoop` proposes calibration or training changes from
  reviewed feedback
- `ContinualEngine` mines recent failures into an adversarial suite and trained
  adversary scorer
- `DefenseRegistry` hot-swaps the active defence only after the review and
  adversarial gates pass

The pipeline does not train, mine, or approve anything by itself. It checks
that an already-created proposal is approved, checks that adversarial mining
produced enough cases, checks the held-out score delta, then performs one
atomic registry promotion with tenant-safe metadata.

```python
from director_ai.core.defense_genome import DefenseRegistry, DefenseUpdatePipeline

registry = DefenseRegistry()
pipeline = DefenseUpdatePipeline(
    registry=registry,
    min_adversarial_cases=8,
    min_holdout_improvement=0.02,
)

report = pipeline.review_and_promote(
    proposal=approved_guard_loop_proposal,
    evolve_report=continual_engine_report,
    defense=candidate_defence,
    version=2,
    label="defence-v2",
    baseline_score=0.72,
    candidate_score=0.84,
)
```

Promotion metadata contains identifiers and metrics only:

- proposal id, proposal type, approval id, manifest id, rollback id
- continual suite version, mined pattern count, adversarial case count
- baseline score, candidate score, and held-out delta

Raw prompts, responses, credentials, private evidence text, and tenant payloads
remain in their owning stores and are not copied into registry metadata.

## Full API

::: director_ai.core.defense_genome.update_pipeline.DefenseUpdatePipeline

::: director_ai.core.defense_genome.update_pipeline.DefenseUpdateReport
