# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory release assembler

"""Assemble the final Customer Model Factory release gate from manifest JSON.

The assembler now lives in the shipped package as
``director_ai._cli_release_gate.assemble_release_gate`` (exposed as
``director-ai release-gate assemble``, WCC-3); this script remains a thin
delegating wrapper for existing tooling and CI invocations.
"""

from __future__ import annotations

from director_ai._cli_release_gate import assemble_release_gate as main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
