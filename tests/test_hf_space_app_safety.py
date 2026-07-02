# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space app safety tests
"""Unit guard for Hugging Face Space app Markdown safety."""

from __future__ import annotations

import importlib
from typing import Protocol, cast


class HfSpaceApp(Protocol):
    """Typed subset of the checked-in Hugging Face Space app."""

    def score_response(
        self,
        facts_text: str,
        query: str,
        llm_response: str,
    ) -> tuple[str, str, str, str]:
        """Score a response through the public Space callback."""


def _load_app_module() -> HfSpaceApp:
    """Import the checked-in Space app through normal Python import resolution."""
    return cast(HfSpaceApp, importlib.import_module("demo.app"))


def test_score_response_escapes_user_controlled_markdown_surfaces() -> None:
    """The Space score callback should escape user-controlled Markdown text."""
    module = _load_app_module()
    malicious_fact = '<img src=x onerror="alert(1)">'

    _badge, details, _bar, context = module.score_response(
        f"payload: {malicious_fact}",
        "payload",
        malicious_fact,
    )

    assert malicious_fact not in details
    assert malicious_fact not in context
    assert "&lt;img" in context
