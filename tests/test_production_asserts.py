# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production assert hardening tests

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDITED_PRODUCTION_FILES = (
    "src/director_ai/core/retrieval/contextual_compression.py",
    "src/director_ai/core/retrieval/query_decomposition.py",
    "src/director_ai/core/scoring/distilled_scorer.py",
    "src/director_ai/core/scoring/embed_scorer.py",
    "src/director_ai/core/scoring/nli.py",
    "src/director_ai/core/scoring/scorer.py",
    "src/director_ai/core/calibration/tuner.py",
)


def test_audited_production_files_do_not_use_runtime_asserts() -> None:
    offenders: list[str] = []
    for relative_path in AUDITED_PRODUCTION_FILES:
        path = PROJECT_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{relative_path}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        )

    assert offenders == []
