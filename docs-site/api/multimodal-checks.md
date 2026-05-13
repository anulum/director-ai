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
- audit payloads and safety events include media references, not raw media,
  transcripts, frame data, or claim text

```python
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.multimodal_guard import (
    MultimodalCheckRequest,
    MultimodalVerifierAdapter,
)

adapter = MultimodalVerifierAdapter(
    image_guard=image_guard,
    enabled_modalities=("image",),
    benchmarked_modalities=("image",),
)

result = adapter.check(
    MultimodalCheckRequest(
        modality="image",
        claim_text="The image shows a labelled package.",
        media_ref="media://image-42",
        image_bytes=image_bytes,
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

## Full API

::: director_ai.core.multimodal_guard.adapter.MultimodalCheckRequest

::: director_ai.core.multimodal_guard.adapter.MultimodalCheckResult

::: director_ai.core.multimodal_guard.adapter.MultimodalVerifierAdapter
