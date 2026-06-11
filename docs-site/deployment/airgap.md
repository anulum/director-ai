<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — airgap deployment guide
-->

# Airgap Install

This example installs the full local safety stack without network access on
the target host: FastAPI service, local Chroma, NLI, ONNX Runtime, UI wizard,
ONNX artefacts, pinned model revisions, and optional Rust kernel wheels.

`requirements/airgap_full_stack.toml` is the tracked manifest for the paths,
models, revisions, and install checks. `scripts/airgap_full_stack_example.sh`
is the shell example.

## Prepare On A Connected Builder

Build or copy the wheelhouse:

```bash
mkdir -p wheelhouse wheelhouse/rust models/factcg-onnx
uv export --locked --extra server --extra vector --extra nli --extra onnx --extra ui \
  --format requirements-txt --output-file /tmp/director-airgap.txt
python -m pip download --dest wheelhouse -r /tmp/director-airgap.txt
```

Add optional Rust wheels:

```bash
python -m pip download --dest wheelhouse/rust backfire-kernel
```

Export ONNX artefacts:

```bash
pip install "director-ai[nli,onnx]"
pip install --no-deps --require-hashes -r requirements/docker-gpu-export.txt
director-ai export --format onnx --output models/factcg-onnx
```

The export environment must keep the repository's `transformers>=5.0.0rc3,<6`
line. Do not install legacy ONNX exporter packages that force
`transformers<5`.

Mirror the pinned model revisions:

| Key | Model | Revision | Target directory |
|-----|-------|----------|------------------|
| `nli` | `yaxili96/FactCG-DeBERTa-v3-Large` | `0430e3509dbd28d2dff7a117c0eae25359ff3e80` | `models/hf/yaxili96/FactCG-DeBERTa-v3-Large` |
| `embedding` | `BAAI/bge-large-en-v1.5` | `d4aa6901d3a41ba39fb536a557fa166f842b0e09` | `models/hf/BAAI/bge-large-en-v1.5` |
| `reranker` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `c5ee24cb16019beea0893ab7796b1df96625c6b8` | `models/hf/cross-encoder/ms-marco-MiniLM-L-6-v2` |

The ONNX directory must contain:

```text
models/factcg-onnx/
  model.onnx
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

Transfer `wheelhouse/`, `models/`, `uv.lock`,
`requirements/airgap_full_stack.toml`, and
`scripts/airgap_full_stack_example.sh` to the target host. The target host must
already have `uv` available; keep the `uv` binary in the same offline operations
tool bundle used to build the wheelhouse.

## Install On The Airgapped Host

Run the tested example:

```bash
bash scripts/airgap_full_stack_example.sh
```

The script performs:

```bash
UV_OFFLINE=1 uv sync --locked --offline --active --extra server --extra vector --extra nli --extra onnx --extra ui
uv pip install --offline --find-links wheelhouse/rust backfire-kernel==0.1.0
director-ai doctor
```

It also sets:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIRECTOR_SCORER_BACKEND=onnx
DIRECTOR_ONNX_PATH=models/factcg-onnx
```

## Verify

Check the installed stack without contacting package indexes:

```bash
director-ai doctor
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python -c "import backfire_kernel; print('rust wheel available')"
```

If the Rust wheel is not present, the Python fallback remains available. Keep
the optional Rust wheelhouse separate so CPU-only targets can omit it.

## Related Files

- `requirements/OPTIONAL_EXTRA_LOCKS.md`
- `requirements/heavy_optional_dependency_policy.toml`
- `requirements/onnx_wheel_targets.toml`
- `docs-site/deployment/onnx-artefacts.md`
- `docs-site/deployment/supply-chain.md`
