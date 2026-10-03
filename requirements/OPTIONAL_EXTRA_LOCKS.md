<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-AI — optional extra lock notes
-->

# Optional Extra Locks

`uv.lock` is the canonical resolved graph for the heavier optional extras:
`[nli]`, `[onnx]`, `[vector]`, `[ui]`, `[server]`, `[physical]`, and
`[enterprise]`.

The policy also pins the `boto3` constraint across `[aws]`, `[ingestion-s3]`,
and `[auto-kb]`: all three declare the identical `>=1.34,<2` range. When one
of them moves, move the other two in the same change — upward only, never by
dropping a cap or lowering a floor (see the 2026-07-16 Dependabot #154
regression for why).

`requirements/pyproject.toml` and `requirements/uv.lock` are intentionally
empty, non-package graph markers for GitHub's uv dependency grapher. The
dependency authority for this directory remains the hashed `*.txt` files and
their checked-in `*.in` sources.

The top-level packages for those extras must keep an upper bound unless
`requirements/uv_extra_lock_policy.toml` records a deliberate exception. The
policy file lists the package names checked by
`tests/test_optional_extra_lock_policy.py`.

Refresh the resolved graph after changing those extras:

```bash
uv lock
```

Install the exact resolved set for a target stack with `--locked`:

```bash
uv sync --locked --extra nli --extra onnx --extra vector --extra ui --extra server --extra enterprise
```

For lighter checks, sync only the extra under review:

```bash
uv sync --locked --extra server
```

The demo extra and both Space README files target Gradio 6.29.1. The
checked-in `requirements/demo.txt` exports the demo dependency graph with
hashes from the same root lock:

```bash
uv export --locked --extra demo --no-dev --no-emit-project \
  --no-emit-package backfire-kernel --no-header \
  --output-file requirements/demo.txt
```

Install it with `--require-hashes`, followed by the application and native
kernel wheels. The live streaming demo also uses the paid UI overlay.
`demo/requirements.txt` is the separate Space package's application/framework
input; the manual Space publication guard checks its framework floor against
the README metadata.

For physical adapters, sync the pinned MuJoCo runtime separately and keep ROS 2
or CARLA in their vendor-managed runtime:

```bash
uv sync --locked --extra physical
```

For zk proof adapters, do not add prover stacks to the default package unless
the adapter is fully pinned and tested. Run arkworks, gnark, snarkjs, or similar
tooling in a separate adapter service, pin circuit artefacts by digest, and
record the circuit id in passports.

Do not widen these extras to open-ended major ranges without updating this
policy and the lockfile in the same change.

Supply-chain controls for heavy optional packages live in
`requirements/heavy_optional_dependency_policy.toml`.

## Shared CI and optional-runtime constraints

Use uv 0.11.33 to regenerate the profiles. CrewAI CLI 1.15.23 requires
`uv~=0.11.6`, so the shared application environment retains the newest
compatible 0.11 release. The dev and CPU inference profiles
are installed together, so their shared packages must resolve compatibly.
`ci-inference.constraints.in` retains Datasets 5.0.1's `fsspec<=2026.6.0`
and `huggingface-hub<2.0` requirements. It also keeps NumPy below 2.5 to
match Presidio Analyzer 2.2.364 in the root optional-extra graph. Both
Presidio extras require that release or newer, preventing a resolver from
selecting an older analyzer to accommodate a newer NumPy. The shared profile
also retains Instructor's `rich<15` bound and CrewAI's `pydantic<2.13`
bound. CrewAI now requires 1.15.23 or later: that upstream release accepts
patched JSON Repair, so the old resolver override
is removed and the upstream dependency contract applies unchanged.

Native hash-locked installation with uv uses the generated profile directly:

```bash
uv --no-config pip install --require-hashes -r requirements/ci-dev.txt
```

The generated profiles already contain the resolved project constraints.
Reapplying the root's unpinned resolver constraints in hash-required installation
mode is invalid. CI uses pip with those same hash-locked profiles.

## CI type-checking dependencies

`requirements/ci-types.in` declares the stubs used by the CI type-check job,
including lxml's DOCX parser exception types. Refresh its hashed installation
file with the existing pins retained:

```bash
uv pip compile requirements/ci-types.in \
  --constraints requirements/ci-types.txt --generate-hashes --no-header \
  --output-file requirements/ci-types.txt
```

The type-check job installs this file with `--require-hashes`. The root dev
extra also declares these stubs for local strict checks.

## Build, fuzz and GPU profiles

The build and fuzz files retain their checked-in native resolver inputs:

```bash
for profile in docker-build ci-build ci-fuzz; do
  uv pip compile "requirements/${profile}.in" --upgrade \
    --python-version 3.11 --universal --generate-hashes --no-header \
    --output-file "requirements/${profile}.txt"
done
```

Docker build tooling includes Maturin 1.15, the kernel's build backend, and
`wheel`, required by the application backend when Docker builds without
isolation. `requirements-dev.txt` delegates to `.[dev]` rather than repeating
older tool floors beside `pyproject.toml`.

Export the selected GPU image runtime from the canonical lock:

```bash
uv export --locked --extra nli --extra server --no-dev \
  --no-emit-project --no-emit-package backfire-kernel \
  --format requirements-txt --output-file requirements/docker-gpu.txt
```

Resolve the ONNX export stage against that runtime, retaining native hashes:

```bash
uv pip compile requirements/docker-gpu-export.in \
  --constraints requirements/docker-gpu.txt --upgrade \
  --python-version 3.11 --universal --generate-hashes --no-header \
  --no-emit-package numpy --no-emit-package packaging \
  --no-emit-package sympy --no-emit-package typing-extensions \
  --output-file requirements/docker-gpu-export.txt
```

Those four omitted dependencies are already installed by the runtime stage.
FlatBuffers remains in the export file because ONNX Runtime requires it.
The export input preserves the shared `protobuf<7` contract; an independently
upgraded protobuf 7.x pin would conflict with the selected gRPC tooling stack.

## Cloud Run CPU profile

`requirements/docker-saas.txt` is the CPython 3.12/Linux AMD64 installation
profile for the Cloud Run image. It selects `[server,nli,onnx,embed]` from the
root lock and replaces CUDA PyTorch with the official CPU wheel of the same
upstream version. The wheel URL and SHA-256 are explicit in
`requirements/docker-saas-cpu.in`; this profile is not portable to Windows,
ARM or another Python minor version.

```bash
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
uv export --locked --no-dev --no-hashes --no-emit-project \
  --no-emit-package backfire-kernel \
  --extra server --extra nli --extra onnx --extra embed \
  --output-file "$constraints"
uv --no-config pip compile pyproject.toml \
  --extra server --extra nli --extra onnx --extra embed \
  --override requirements/docker-saas-cpu.in --constraint "$constraints" \
  --no-emit-package backfire-kernel --no-sources --generate-hashes \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --no-header --no-annotate --output-file requirements/docker-saas.txt
```

The image installs both application wheels with the checked-in build hooks,
exports the pinned FactCG model without quantisation, and loads that artefact
offline. See the [Cloud Run deployment guide](../docs-site/deployment/cloud-run.md).

The Type Check CI job runs Python 3.11 to match the repository's configured
MyPy language floor. Newer NumPy and Deprecated distributions can contain
Python 3.12 type statements; run the floor check with the corresponding 3.11
dependency selection. The Python 3.11–3.13 runtime test matrix still exercises
the marker-selected packages for each supported interpreter.
