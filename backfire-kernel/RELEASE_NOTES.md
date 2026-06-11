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
