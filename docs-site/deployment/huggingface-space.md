# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space deployment

# Hugging Face Space

The hosted Gradio demo is packaged from the `demo/` directory. The Space
repository receives exactly three root files:

| Space file | Source |
|------------|--------|
| `app.py` | `demo/app.py` |
| `requirements.txt` | `demo/requirements.txt` |
| `README.md` | `demo/README_HF.md` |

The package is described by `demo/hf_space_manifest.toml` and checked by
`tools/validate_hf_space_demo.py`. The validator enforces the Space front
matter, the Gradio and package requirements, an importable `build_app()`
entrypoint, and explicit staging in the publish script.

## Local Validation

Run the Space package gate before publishing:

```bash
uv run --frozen python tools/validate_hf_space_demo.py .
uv run --frozen pytest -q tests/test_hf_space_demo.py tests/test_demo_streaming_halt_live.py
```

The app must stay credential-free. The demo uses deterministic local examples
and should not require secrets, account tokens, or private datasets at runtime.

## Publish Procedure

Authenticate with the Hugging Face CLI, then run the guarded copy script:

```bash
hf auth login
bash demo/push_to_hf.sh
```

The script clones the Space into a temporary directory, copies only the declared
files, stages only `app.py`, `requirements.txt`, and `README.md`, and asks for a
manual confirmation before pushing. Abort if the staged diff contains anything
outside that set.

## Platform Contract

Hugging Face Spaces read the Space configuration from the YAML block at the top
of the repository root `README.md`. The root `requirements.txt` supplies Python
dependencies for Gradio Spaces.

References:

- [Spaces configuration reference](https://huggingface.co/docs/hub/en/spaces-config-reference)
- [Spaces dependency handling](https://huggingface.co/docs/hub/spaces-dependencies)
