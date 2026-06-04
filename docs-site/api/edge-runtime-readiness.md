<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — edge runtime readiness API
-->

# Edge Runtime Readiness

`build_edge_runtime_readiness()` records the current edge/mobile deployment
state for browser, Worker, embedded, and local low-latency paths. It checks the
tracked WASM release plan, Rust kernel sources, ONNX and quantisation contracts,
local build artefacts, and smoke-test evidence without running external builds.

Use it when preparing a release packet or deciding whether the edge path is only
ready for local trial or ready for release:

```python
from director_ai.core.edge import build_edge_runtime_readiness

profile = build_edge_runtime_readiness(
    ".",
    target_id="browser-worker",
    quantised_model_path="MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
    browser_smoke_evidence="benchmarks/results/browser-worker-smoke.json",
    mobile_smoke_evidence="benchmarks/results/mobile-smoke.json",
)

assert profile.ready_for_local_trial
```

`ready_for_local_trial` means the tracked source contracts, target matrix,
deployment docs, and latency-benchmark surfaces exist. `ready_for_release` is
stricter: it also requires local WASM build artefacts, a quantised model
artefact, Rust Python accelerator import evidence, and browser/mobile smoke
evidence.

Generate the local R14 evidence packet with:

```bash
PYTHONPATH=src python -m benchmarks.edge_mobile_evidence
```

The packet is tenant-safe. Paths outside the repository are recorded as an
external path marker instead of serialising absolute local paths.

## Full API

::: director_ai.core.edge.runtime_profile.EdgeRuntimeCheck

::: director_ai.core.edge.runtime_profile.EdgeRuntimeReadiness

::: director_ai.core.edge.runtime_profile.build_edge_runtime_readiness
