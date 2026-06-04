# Multimodal Checks

Multimodal checks adapt image, audio, and video evidence into the shared
`GuardDecision` and `SafetyEvent` contracts. The adapter is explicitly opt-in:
a modality must be enabled before it can be checked, and a modality must be
marked benchmarked before a supported result may become `allow`.

## Decision Boundary

`MultimodalVerifierAdapter` enforces production-safe defaults:

- disabled or unsupported modalities raise errors instead of silently passing
- uncertain evidence maps to `warn`, never `allow`
- hallucinated or temporally inconsistent evidence maps to `halt`
- unbenchmarked modalities map to `warn` even when the low-level checker says
  the claim is consistent
- optional caption and metadata grounding can reduce a modality score before a
  decision is emitted
- audit payloads and safety events include media references, not raw media,
  transcripts, frame data, captions, metadata values, or claim text

```python
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.multimodal_guard import (
    MultimodalCheckRequest,
    MultimodalVerifierAdapter,
)

adapter = MultimodalVerifierAdapter(
    image_guard=image_guard,
    caption_score_fn=caption_grounder,
    metadata_score_fn=metadata_grounder,
    enabled_modalities=("image",),
    benchmarked_modalities=("image",),
)

result = adapter.check(
    MultimodalCheckRequest(
        modality="image",
        claim_text="The image shows a labelled package.",
        media_ref="media://image-42",
        image_bytes=image_bytes,
        caption_text="Package label is absent.",
        metadata={"captured_at": "2026-05-13", "source": "inspection-rig"},
    ),
    risk_envelope=RiskEnvelope(
        action_category="multimodal",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    ),
    policy_id="policy.multimodal.regulated",
)
```

Grounding callbacks receive either `(caption_text, claim_text)` or
`(metadata, claim_text)` and must return a finite score in `[0, 1]`. Scores
below the grounding floor halt the claim; scores below the grounding allow
threshold produce a warning unless the base verifier already found a stricter
verdict. Evidence references use suffixes such as `#caption` and
`#metadata:captured_at`, so downstream audit logs can identify which grounding
channel was used without storing private captions or metadata values.

## Local Evidence Packet

Generate the local multimodal temporal evidence packet before promoting a
deployment that enables image, audio, or video checks:

```bash
PYTHONPATH=src python -m benchmarks.multimodal_temporal_evidence
```

The packet exercises image allow/halt paths, caption-grounding conflicts, video
frame temporal halts, and the dependency-free hash-bag encoder/verifier path. It
does not include an external Vision-NLI benchmark or a real video model run.

## Full API

::: director_ai.core.multimodal_guard.adapter.MultimodalCheckRequest

::: director_ai.core.multimodal_guard.adapter.MultimodalCheckResult

::: director_ai.core.multimodal_guard.adapter.MultimodalVerifierAdapter
