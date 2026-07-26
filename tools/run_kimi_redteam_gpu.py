# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Deprecated command shim for the exact KIMI audit-reproduction workflow.

The reusable implementation has moved to ``run_remote_benchmark_gpu.py``.
This file preserves the historical command path and its KIMI-specific defaults
for audit reproducibility; new automation should use the neutral runner and
pass ``--script`` explicitly.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from importlib import import_module
from typing import cast

_implementation = (
    "tools.run_remote_benchmark_gpu" if __package__ else "run_remote_benchmark_gpu"
)
_main = cast(
    Callable[[list[str] | None], int],
    import_module(_implementation).main,
)

logger = logging.getLogger("DirectorAI.KimiAuditCompatibility")


def _legacy_argv(argv: list[str]) -> list[str]:
    """Supply historical audit defaults without weakening the neutral CLI."""
    translated = list(argv)
    if "--script" not in translated:
        translated.extend(["--script", "benchmarks/kimi_redteam_reproduction.py"])
    if "--out" not in translated:
        translated.extend(["--out", "kimi_redteam_reproduction.json"])
    return translated


def main(argv: list[str] | None = None) -> int:
    """Run the neutral runner with the historical KIMI audit defaults."""
    logger.warning(
        "tools/run_kimi_redteam_gpu.py is deprecated; use "
        "tools/run_remote_benchmark_gpu.py --script ..."
    )
    return _main(_legacy_argv(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
