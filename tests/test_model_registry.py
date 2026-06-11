# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — fallback model registry tests
"""Multi-angle tests for the fallback model registry.

Covers the built-in NLI chain (every entry revision-pinned), the candidate
ordering (primary first, fallbacks resolved, primary not duplicated), resolution
(primary available → primary; primary delisted → first available fallback; all
unavailable → primary unchanged; first-available-wins ordering), probe caching,
the unknown-role passthrough, the unpinned-chain construction guard, and the
DirectorConfig wiring (default-off uses the primary with no probe; enabled
resolves through the registry). An injected probe keeps every case offline.
"""

from __future__ import annotations

import pytest

from director_ai.core.model_registry import (
    FALLBACK_CHAINS,
    FallbackModelRegistry,
    ResolvedModel,
)
from director_ai.core.model_revisions import resolve_model_revision

_PRIMARY = "yaxili96/FactCG-DeBERTa-v3-Large"
_PRIMARY_REV = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
_FALLBACK1 = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def _always(value):
    return lambda model_id, revision: value


class TestChain:
    def test_nli_chain_present(self):
        assert "nli" in FALLBACK_CHAINS
        assert _FALLBACK1 in FALLBACK_CHAINS["nli"]

    def test_every_chain_entry_is_pinned(self):
        for models in FALLBACK_CHAINS.values():
            for model_id in models:
                assert resolve_model_revision(model_id) is not None


class TestCandidates:
    def test_primary_first_then_fallbacks(self):
        reg = FallbackModelRegistry(probe=_always(True))
        cands = reg.candidates("nli", _PRIMARY, primary_revision=_PRIMARY_REV)
        assert cands[0] == (_PRIMARY, _PRIMARY_REV)
        assert cands[1][0] == _FALLBACK1
        assert cands[1][1] == resolve_model_revision(_FALLBACK1)

    def test_primary_not_duplicated_when_in_chain(self):
        reg = FallbackModelRegistry(probe=_always(True))
        cands = reg.candidates("nli", _FALLBACK1)
        assert [c[0] for c in cands].count(_FALLBACK1) == 1

    def test_unknown_role_yields_only_primary(self):
        reg = FallbackModelRegistry(probe=_always(True))
        assert reg.candidates("embedding", _PRIMARY) == [(_PRIMARY, None)]


class TestResolve:
    def test_primary_available_uses_primary(self):
        res = FallbackModelRegistry(probe=_always(True)).resolve(
            "nli", _PRIMARY, primary_revision=_PRIMARY_REV
        )
        assert res == ResolvedModel(_PRIMARY, _PRIMARY_REV, "nli", is_fallback=False)

    def test_primary_delisted_falls_back(self):
        reg = FallbackModelRegistry(probe=lambda m, r: "FactCG" not in m)
        res = reg.resolve("nli", _PRIMARY, primary_revision=_PRIMARY_REV)
        assert res.model_id == _FALLBACK1
        assert res.is_fallback
        assert res.revision == resolve_model_revision(_FALLBACK1)

    def test_all_unavailable_keeps_primary(self):
        res = FallbackModelRegistry(probe=_always(False)).resolve(
            "nli", _PRIMARY, primary_revision=_PRIMARY_REV
        )
        assert res.model_id == _PRIMARY
        assert not res.is_fallback

    def test_first_available_fallback_wins(self):
        # Primary and the first fallback are down; the second is reachable.
        order = FALLBACK_CHAINS["nli"]
        reg = FallbackModelRegistry(probe=lambda m, r: m == order[1])
        res = reg.resolve("nli", _PRIMARY)
        assert res.model_id == order[1]
        assert res.is_fallback

    def test_probe_result_is_cached(self):
        calls = []

        def probe(model_id, revision):
            calls.append(model_id)
            return True

        reg = FallbackModelRegistry(probe=probe)
        reg.resolve("nli", _PRIMARY)
        reg.resolve("nli", _PRIMARY)
        assert calls.count(_PRIMARY) == 1


class TestConstructionGuard:
    def test_unpinned_chain_entry_raises(self):
        with pytest.raises(ValueError, match="not"):
            FallbackModelRegistry(
                probe=_always(True), chains={"nli": ("some/unpinned-model-xyz",)}
            )

    def test_custom_pinned_chain_accepted(self):
        reg = FallbackModelRegistry(
            probe=_always(True), chains={"nli": ("roberta-large-mnli",)}
        )
        assert reg.resolve("nli", _PRIMARY, primary_revision=_PRIMARY_REV).model_id == (
            _PRIMARY
        )


class TestConfigWiring:
    def test_default_off_uses_primary_without_probe(self, monkeypatch):
        import director_ai.core.model_registry as mr

        monkeypatch.setattr(
            mr, "_hub_availability", lambda m, r: pytest.fail("probed when off")
        )
        captured = self._capture_nli_model(monkeypatch)
        from director_ai.core.config import DirectorConfig

        DirectorConfig(use_nli=True).build_scorer()
        assert captured["nli_model"] == _PRIMARY

    def test_enabled_falls_back_on_delisted_primary(self, monkeypatch):
        import director_ai.core.model_registry as mr

        monkeypatch.setattr(mr, "_hub_availability", lambda m, r: "FactCG" not in m)
        captured = self._capture_nli_model(monkeypatch)
        from director_ai.core.config import DirectorConfig

        DirectorConfig(use_nli=True, model_fallback_enabled=True).build_scorer()
        assert captured["nli_model"] == _FALLBACK1

    @staticmethod
    def _capture_nli_model(monkeypatch):
        import director_ai.core.scoring.scorer as scmod

        captured: dict[str, object] = {}
        original = scmod.CoherenceScorer.__init__

        def spy(self, *args, **kwargs):
            captured["nli_model"] = kwargs.get("nli_model")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(scmod.CoherenceScorer, "__init__", spy)
        return captured
