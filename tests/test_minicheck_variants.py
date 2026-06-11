# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MiniCheck variant + precision-tier tests
"""Tests for the MiniCheck variant selector and the precision-tier profiles.

Covers the `NLIScorer` variant validation and the variant→checkpoint map, the
pinned immutable revisions for the Flan-T5-Large and Bespoke-MiniCheck-7B
checkpoints, the three `minicheck-*` profiles (backend / variant / dtype /
8-bit), their operator metadata, and the wiring that threads the variant from
`DirectorConfig.build_scorer` through `CoherenceScorer` into the NLI scorer and
the lazy summarisation MiniCheck scorer. No model is downloaded — the variant
plumbing is exercised with `use_model=False`.
"""

from __future__ import annotations

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.model_revisions import resolve_model_revision
from director_ai.core.scoring.nli import NLIScorer

_VARIANTS = ("deberta-v3-large", "flan-t5-large", "Bespoke-MiniCheck-7B")


class TestVariantSelection:
    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_valid_variants_accepted(self, variant):
        scorer = NLIScorer(
            use_model=False, backend="minicheck", minicheck_variant=variant
        )
        assert scorer._minicheck_variant == variant

    def test_default_variant_is_deberta(self):
        assert NLIScorer(use_model=False)._minicheck_variant == "deberta-v3-large"

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="minicheck_variant must be one of"):
            NLIScorer(backend="minicheck", minicheck_variant="gpt-9")

    def test_ckpt_map_covers_every_variant(self):
        for variant in _VARIANTS:
            assert variant in NLIScorer._MINICHECK_CKPTS
        assert (
            NLIScorer._MINICHECK_CKPTS["Bespoke-MiniCheck-7B"]
            == "bespokelabs/Bespoke-MiniCheck-7B"
        )


class TestPinnedRevisions:
    @pytest.mark.parametrize(
        "model_id, sha",
        [
            (
                "bespokelabs/Bespoke-MiniCheck-7B",
                "1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8",
            ),
            (
                "lytang/MiniCheck-Flan-T5-Large",
                "96eafd01cee2d16cf81aaa2fb226b14f422a37b3",
            ),
            (
                "lytang/MiniCheck-DeBERTa-v3-Large",
                "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb",
            ),
        ],
    )
    def test_minicheck_checkpoints_are_pinned(self, model_id, sha):
        assert resolve_model_revision(model_id) == sha

    def test_every_mapped_ckpt_is_pinned(self):
        for ckpt in NLIScorer._MINICHECK_CKPTS.values():
            # Each MiniCheck checkpoint resolves to an immutable revision.
            assert resolve_model_revision(ckpt) is not None


class TestPrecisionProfiles:
    def test_fast_profile_is_fp16_deberta(self):
        cfg = DirectorConfig.from_profile("minicheck-fast")
        assert cfg.scorer_backend == "minicheck"
        assert cfg.minicheck_variant == "deberta-v3-large"
        assert cfg.nli_torch_dtype == "float16"
        assert cfg.use_nli is True

    def test_balanced_profile_is_full_precision_deberta(self):
        cfg = DirectorConfig.from_profile("minicheck-balanced")
        assert cfg.minicheck_variant == "deberta-v3-large"
        assert cfg.nli_torch_dtype == ""
        assert cfg.nli_quantize_8bit is False

    def test_accurate_profile_is_8bit_7b(self):
        cfg = DirectorConfig.from_profile("minicheck-accurate")
        assert cfg.minicheck_variant == "Bespoke-MiniCheck-7B"
        assert cfg.nli_quantize_8bit is True

    @pytest.mark.parametrize(
        "name", ["minicheck-fast", "minicheck-balanced", "minicheck-accurate"]
    )
    def test_profile_metadata_present(self, name):
        meta = DirectorConfig.profile_metadata(name)
        assert meta.name == name
        assert "minicheck" in meta.required_dependencies

    def test_accurate_profile_declares_bitsandbytes(self):
        meta = DirectorConfig.profile_metadata("minicheck-accurate")
        assert "bitsandbytes" in meta.required_dependencies


class TestScorerWiring:
    def test_config_threads_variant_into_scorer(self):
        scorer = DirectorConfig.from_profile("minicheck-accurate").build_scorer()
        assert scorer._minicheck_variant == "Bespoke-MiniCheck-7B"

    def test_coherence_scorer_stores_variant(self):
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False, minicheck_variant="flan-t5-large")
        assert scorer._minicheck_variant == "flan-t5-large"

    def test_lazy_minicheck_scorer_uses_variant(self, monkeypatch):
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False, minicheck_variant="flan-t5-large")
        captured = {}
        real_init = NLIScorer.__init__

        def _spy(self, *args, **kwargs):
            captured["variant"] = kwargs.get("minicheck_variant")
            real_init(self, *args, **kwargs)
            # Avoid touching the real package; report "loaded".
            monkeypatch.setattr(self, "_ensure_minicheck", lambda: True)

        monkeypatch.setattr(NLIScorer, "__init__", _spy)
        scorer._get_minicheck_scorer()
        assert captured["variant"] == "flan-t5-large"
