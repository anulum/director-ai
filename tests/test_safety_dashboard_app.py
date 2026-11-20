# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# DIRECTOR-AI — tests/test_safety_dashboard_app.py

"""Contract tests for the safety-dashboard Gradio app module.

``director_ai.ui._dashboard_app`` owns the interactive UI shell of the
safety operations dashboard; ``safety_dashboard`` re-exports its
launcher unchanged. These tests pin where the launcher lives and the
facade re-export seam the CLI patches; the widget wiring matrix (fake
Gradio) stays in ``tests/test_safety_dashboard.py``.
"""

from __future__ import annotations

import director_ai.ui._dashboard_app as dashboard_app_module
from director_ai.ui import safety_dashboard


class TestModulePlacement:
    """Protect the dashboard launcher import boundary."""

    def test_launcher_is_defined_in_the_app_module(self) -> None:
        """Keep the launcher implementation in the application module."""
        launcher = dashboard_app_module.launch_safety_dashboard
        assert launcher.__module__ == dashboard_app_module.__name__

    def test_facade_re_exports_the_same_launcher_object(self) -> None:
        """Preserve launcher identity through the dashboard facade."""
        assert (
            safety_dashboard.launch_safety_dashboard
            is dashboard_app_module.launch_safety_dashboard
        )

    def test_module_exports_only_the_launcher(self) -> None:
        """Keep the application module export list limited to its launcher."""
        assert dashboard_app_module.__all__ == ["launch_safety_dashboard"]

    def test_app_module_has_no_import_time_builder_or_gradio_binding(self) -> None:
        """Import the application module without loading Gradio bindings or its builder."""
        for name in ("build_safety_dashboard", "gr", "gradio"):
            assert not hasattr(dashboard_app_module, name)
