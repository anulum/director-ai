# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for hardened mode and auto scorer backend selection.

Covers hardened mode flag propagation, auto backend resolution,
and combined config behaviour.
"""

from __future__ import annotations

from director_ai.core.config import DirectorConfig

# ── Hardened mode ──────────────────────────────────────────────────────


class TestHardenedMode:
    def test_default_off(self):
        cfg = DirectorConfig()
        assert cfg.hardened is False

    def test_enables_all_safety(self):
        cfg = DirectorConfig(hardened=True, api_keys='["sk-test"]')
        assert cfg.production_mode is True
        assert cfg.use_nli is True
        assert cfg.injection_detection_enabled is True
        assert cfg.sanitize_inputs is True
        assert cfg.redact_pii is True

    def test_requires_auth_via_production_mode(self):
        import pytest

        with pytest.raises(ValueError, match="api_keys"):
            DirectorConfig(hardened=True)

    def test_hardened_with_api_keys(self):
        cfg = DirectorConfig(hardened=True, api_keys='["sk-test"]')
        assert cfg.hardened is True


# ── Auto scorer backend ───────────────────────────────────────────────


class TestAutoScorer:
    def test_default_is_auto(self):
        cfg = DirectorConfig()
        assert cfg.scorer_backend == "auto"

    def test_resolve_auto_picks_something(self):
        cfg = DirectorConfig(use_nli=False)
        backend = cfg._resolve_scorer_backend()
        # Auto picks best available — rust if backfire installed, else lite
        assert backend in ("rust", "lite", "deberta", "onnx")

    def test_resolve_with_nli(self):
        cfg = DirectorConfig(use_nli=True)
        backend = cfg._resolve_scorer_backend()
        # Should pick rust (if available) or deberta
        assert backend in ("rust", "deberta")

    def test_resolve_with_onnx_path(self):
        cfg = DirectorConfig(use_nli=True, onnx_path="/tmp/model.onnx")
        backend = cfg._resolve_scorer_backend()
        # Should pick rust (if available) or onnx
        assert backend in ("rust", "onnx")

    def test_explicit_backend_not_overridden(self):
        cfg = DirectorConfig(scorer_backend="rules")
        assert cfg._resolve_scorer_backend() == "rules"

    def test_explicit_lite(self):
        cfg = DirectorConfig(scorer_backend="lite")
        assert cfg._resolve_scorer_backend() == "lite"
