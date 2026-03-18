# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI -- Import Guard Tests

# -----------------------------------------------------------------------
# Director-Class AI -- Import Guard Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# -----------------------------------------------------------------------
"""Verify that heavyweight optional dependencies are not imported eagerly."""

import ast
from pathlib import Path


def test_scorer_no_eager_torch_import():
    """Verify scorer.py does not import torch at module level."""
    scorer_path = (
        Path(__file__).parent.parent / "src" / "director_ai" / "core" / "scorer.py"
    )
    tree = ast.parse(scorer_path.read_text())

    top_level_imports = [
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node.col_offset == 0
    ]
    assert "torch" not in top_level_imports, (
        "torch must not be imported at module level in scorer.py"
    )


def test_scorer_no_eager_transformers_import():
    """Verify scorer.py does not import transformers at module level."""
    scorer_path = (
        Path(__file__).parent.parent / "src" / "director_ai" / "core" / "scorer.py"
    )
    tree = ast.parse(scorer_path.read_text())

    top_level_from_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.col_offset == 0 and node.module
    ]
    assert "transformers" not in top_level_from_imports, (
        "transformers must not be imported at module level in scorer.py"
    )
