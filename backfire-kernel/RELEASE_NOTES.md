<!--
SPDX-License-Identifier: Apache-2.0
Commercial licence available
Concepts 1996-2026 Miroslav Sotek. All rights reserved.
Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI - Backfire kernel release notes
-->

# Backfire Kernel Release Notes

## 0.1.3 - 2026-06-21

- Built the Python wheel against the PyO3 `abi3-py311` limited API, so a single
  `cp311-abi3` wheel per platform supports Python 3.11, 3.12 and 3.13. This
  closes the install gap where the previous CPython-3.12-only wheels forced
  3.11/3.13 users to build the crate from the source distribution.

## 0.1.2 - 2026-06-21

- Added accelerated Rust paths: span token-to-span reduction, Bulletproof
  range-proof attestation, informal-fallacy detection and chain-of-thought
  arithmetic verification.
- Migrated the bindings to PyO3 0.29.

## 0.1.1 - 2026-04-30

- Aligned Rust workspace and Python wheel versions at `0.1.1`.
- Added a Python module `__version__` export sourced from the crate package
  version.
- Added a release contract at `requirements/backfire_kernel_release.toml`.
- Added the standalone advanced runtime CI entrypoint at
  `backfire-kernel/ci/advanced_runtime_ci.sh`.
- Kept `director-ai[rust]` pinned to the supported `>=0.1.0,<0.2` Python wheel range while the Rust workspace version remains `0.1.1`.

## 0.1.0 - 2026-04-12

- Initial PyO3 wheel boundary for the optional Rust acceleration path.
