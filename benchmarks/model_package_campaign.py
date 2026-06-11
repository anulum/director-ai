# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — model package campaign compatibility entrypoint

"""Provider-neutral entrypoint for model package benchmark campaigns."""

from __future__ import annotations

from benchmarks.model_package_vertex_campaign import main

if __name__ == "__main__":
    raise SystemExit(main())
