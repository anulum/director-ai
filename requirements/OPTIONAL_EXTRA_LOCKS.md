<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — optional extra lock notes
-->

# Optional Extra Locks

`uv.lock` is the canonical resolved graph for the heavier optional extras:
`[nli]`, `[onnx]`, `[vector]`, `[ui]`, `[server]`, `[physical]`, and
`[enterprise]`.

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
