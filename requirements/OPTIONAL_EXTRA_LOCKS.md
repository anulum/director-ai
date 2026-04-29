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
`[nli]`, `[onnx]`, `[vector]`, `[ui]`, `[server]`, and `[enterprise]`.

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

Do not widen these extras to open-ended major ranges without updating this
policy and the lockfile in the same change.
