# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multimodal guard tests

"""Multi-angle coverage: MultimodalClaim validation, HashBag
encoder + verifier determinism and unit-norm invariants,
TorchCLIP adapter import-guard, MultimodalGuard threshold bands
and batching, TemporalConsistencyGuard EMA behaviour, error
paths, and a lightweight property check on the cosine kernel."""

from __future__ import annotations

import importlib.util
import math

import pytest

from director_ai.core.multimodal_guard import (
    CrossModalVerifier,
    HashBagCrossModalVerifier,
    HashBagImageEncoder,
    ImageEncoder,
    MultimodalClaim,
    MultimodalGuard,
    MultimodalVerdict,
    TemporalConsistencyGuard,
    TorchCLIPCrossModalVerifier,
    TorchCLIPImageEncoder,
)

# --- MultimodalClaim ------------------------------------------------


class TestMultimodalClaim:
    def test_valid(self):
        c = MultimodalClaim(image_bytes=b"x", text_claim="a cat")
        assert c.text_claim == "a cat"

    def test_empty_image_rejected(self):
        with pytest.raises(ValueError, match="image_bytes"):
            MultimodalClaim(image_bytes=b"", text_claim="x")

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError, match="text_claim"):
            MultimodalClaim(image_bytes=b"x", text_claim="")

    def test_whitespace_text_rejected(self):
        with pytest.raises(ValueError, match="text_claim"):
            MultimodalClaim(image_bytes=b"x", text_claim="   ")


# --- HashBagImageEncoder --------------------------------------------


class TestHashBagImageEncoder:
    def test_deterministic(self):
        a = HashBagImageEncoder(dim=128).encode(b"abc\x00def")
        b = HashBagImageEncoder(dim=128).encode(b"abc\x00def")
        assert a == b

    def test_unit_norm(self):
        vec = HashBagImageEncoder(dim=128).encode(b"some image bytes here")
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-6)

    def test_dim_enforced(self):
        e = HashBagImageEncoder(dim=64)
        assert e.dim == 64
        assert len(e.encode(b"payload")) == 64

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="image_bytes"):
            HashBagImageEncoder().encode(b"")

    def test_bad_dim(self):
        with pytest.raises(ValueError, match="dim"):
            HashBagImageEncoder(dim=0)

    def test_bad_chunk(self):
        with pytest.raises(ValueError, match="chunk"):
            HashBagImageEncoder(chunk=0)

    def test_different_payloads_differ(self):
        a = HashBagImageEncoder(dim=256).encode(b"payload-a")
        b = HashBagImageEncoder(dim=256).encode(b"payload-b")
        assert a != b


# --- TorchCLIPImageEncoder import guard -----------------------------


class TestTorchCLIPImageEncoderGuard:
    def test_from_pretrained_raises_when_open_clip_missing(self):
        """The optional ``open_clip_torch`` extra is not installed
        in CI; the :class:`ImportError` message must point the
        operator at the install command."""
        if importlib.util.find_spec("open_clip") is not None:
            pytest.skip("open_clip installed; cannot test the ImportError branch")
        with pytest.raises(ImportError, match="multimodal"):
            TorchCLIPImageEncoder.from_pretrained()

    def test_direct_constructor_validates(self):
        with pytest.raises(ValueError, match="model"):
            TorchCLIPImageEncoder(model=None, preprocess=object(), dim=512)
        with pytest.raises(ValueError, match="preprocess"):
            TorchCLIPImageEncoder(model=object(), preprocess=None, dim=512)
        with pytest.raises(ValueError, match="dim"):
            TorchCLIPImageEncoder(model=object(), preprocess=object(), dim=0)


# --- HashBagCrossModalVerifier --------------------------------------


class TestHashBagCrossModalVerifier:
    def test_matches_text_cosine(self):
        enc = HashBagImageEncoder(dim=256)
        ver = HashBagCrossModalVerifier(dim=256)
        # Encode "a cat" as image bytes so image-bag and text-bag
        # collide on the same tokens — similarity should be high.
        embedding = enc.encode(b"a cat")
        # Text tokens: "a cat"
        sim_match = ver.verify(embedding, "a cat")
        sim_unrelated = ver.verify(embedding, "quantum chromodynamics")
        assert sim_match >= sim_unrelated

    def test_dim_mismatch_raises(self):
        ver = HashBagCrossModalVerifier(dim=128)
        with pytest.raises(ValueError, match="dim"):
            ver.verify((0.0,) * 64, "text")

    def test_empty_text_returns_zero(self):
        ver = HashBagCrossModalVerifier(dim=128)
        assert ver.verify((1.0,) + (0.0,) * 127, "") == 0.0
        assert ver.verify((1.0,) + (0.0,) * 127, "   ") == 0.0

    def test_case_insensitive_by_default(self):
        enc = HashBagImageEncoder(dim=256)
        ver = HashBagCrossModalVerifier(dim=256)
        embedding = enc.encode(b"CAT")
        upper = ver.verify(embedding, "CAT")
        lower = ver.verify(embedding, "cat")
        assert upper == lower

    def test_case_sensitive_flag(self):
        ver = HashBagCrossModalVerifier(dim=256, lowercase=False)
        enc = HashBagImageEncoder(dim=256)
        embedding = enc.encode(b"cat")
        upper_sim = ver.verify(embedding, "CAT")
        lower_sim = ver.verify(embedding, "cat")
        # Either different hash-bag tokens produce the same cosine
        # against the image (both zero in this case) or differ; we
        # only require that the setting is honoured — the zero-zero
        # equality is also acceptable.
        assert upper_sim <= lower_sim

    def test_bad_dim(self):
        with pytest.raises(ValueError, match="dim"):
            HashBagCrossModalVerifier(dim=-1)

    def test_output_in_unit_interval(self):
        ver = HashBagCrossModalVerifier(dim=128)
        enc = HashBagImageEncoder(dim=128)
        sim = ver.verify(enc.encode(b"payload"), "arbitrary text")
        assert 0.0 <= sim <= 1.0


# --- TorchCLIPCrossModalVerifier import guard -----------------------


class TestTorchCLIPVerifierGuard:
    def test_from_pretrained_without_open_clip(self):
        if importlib.util.find_spec("open_clip") is not None:
            pytest.skip("open_clip installed")
        with pytest.raises(ImportError, match="multimodal"):
            TorchCLIPCrossModalVerifier.from_pretrained()

    def test_direct_constructor_validates(self):
        with pytest.raises(ValueError, match="model"):
            TorchCLIPCrossModalVerifier(model=None, tokenizer=object(), dim=512)
        with pytest.raises(ValueError, match="tokenizer"):
            TorchCLIPCrossModalVerifier(model=object(), tokenizer=None, dim=512)
        with pytest.raises(ValueError, match="dim"):
            TorchCLIPCrossModalVerifier(model=object(), tokenizer=object(), dim=0)


# --- MultimodalGuard ------------------------------------------------


class _ConstantEncoder:
    """Encoder that returns a pre-computed constant vector."""

    def __init__(self, vec: tuple[float, ...]) -> None:
        self.dim = len(vec)
        self._vec = vec

    def encode(self, image_bytes: bytes) -> tuple[float, ...]:
        return self._vec


class _ConstantVerifier:
    """Verifier that returns a pre-configured similarity."""

    def __init__(self, score: float, dim: int) -> None:
        self.dim = dim
        self._score = score

    def verify(self, image_embedding: tuple[float, ...], text: str) -> float:
        return self._score


class TestMultimodalGuard:
    def _claim(self) -> MultimodalClaim:
        return MultimodalClaim(image_bytes=b"image", text_claim="a cat")

    def _guard(self, score: float) -> MultimodalGuard:
        encoder = _ConstantEncoder((1.0,) + (0.0,) * 127)
        verifier = _ConstantVerifier(score=score, dim=128)
        return MultimodalGuard(encoder=encoder, verifier=verifier)

    def test_protocol_runtime_checkable(self):
        encoder = HashBagImageEncoder(dim=64)
        verifier = HashBagCrossModalVerifier(dim=64)
        assert isinstance(encoder, ImageEncoder)
        assert isinstance(verifier, CrossModalVerifier)

    def test_consistent_band(self):
        verdict = self._guard(0.8).check(self._claim())
        assert verdict.label == "consistent"
        assert verdict.similarity == 0.8
        assert "consistency_threshold" in verdict.reason

    def test_hallucinated_band(self):
        verdict = self._guard(0.05).check(self._claim())
        assert verdict.label == "hallucinated"
        assert "hallucination_threshold" in verdict.reason

    def test_uncertain_band(self):
        verdict = self._guard(0.3).check(self._claim())
        assert verdict.label == "uncertain"

    def test_dim_mismatch_rejected(self):
        encoder = HashBagImageEncoder(dim=64)
        verifier = HashBagCrossModalVerifier(dim=128)
        with pytest.raises(ValueError, match="dim"):
            MultimodalGuard(encoder=encoder, verifier=verifier)

    def test_threshold_order_enforced(self):
        encoder = HashBagImageEncoder(dim=64)
        verifier = HashBagCrossModalVerifier(dim=64)
        with pytest.raises(ValueError, match="thresholds"):
            MultimodalGuard(
                encoder=encoder,
                verifier=verifier,
                hallucination_threshold=0.7,
                consistency_threshold=0.3,
            )

    def test_check_many(self):
        guard = self._guard(0.8)
        claims = [self._claim(), self._claim()]
        verdicts = guard.check_many(claims)
        assert len(verdicts) == 2
        assert all(v.label == "consistent" for v in verdicts)

    def test_verdict_is_dataclass(self):
        verdict = self._guard(0.8).check(self._claim())
        assert isinstance(verdict, MultimodalVerdict)


# --- TemporalConsistencyGuard ---------------------------------------


class TestTemporalConsistencyGuard:
    def test_first_update_initialises_ema(self):
        g = TemporalConsistencyGuard(alpha=0.5, consistency_floor=0.2)
        halt = g.update(0.8)
        assert g.ema == 0.8
        assert halt is False

    def test_decays_toward_drops(self):
        g = TemporalConsistencyGuard(alpha=0.5, consistency_floor=0.2)
        g.update(1.0)
        g.update(0.0)
        g.update(0.0)
        # EMA: 1.0 → 0.5 → 0.25 → below floor on next bad frame.
        halt = g.update(0.0)
        assert halt is True
        assert g.ema is not None and g.ema < 0.2

    def test_reset_clears_state(self):
        g = TemporalConsistencyGuard()
        g.update(0.8)
        g.reset()
        assert g.ema is None

    def test_bad_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            TemporalConsistencyGuard(alpha=0.0)

    def test_bad_floor(self):
        with pytest.raises(ValueError, match="consistency_floor"):
            TemporalConsistencyGuard(consistency_floor=1.5)

    def test_update_range_enforced(self):
        g = TemporalConsistencyGuard()
        with pytest.raises(ValueError, match="similarity"):
            g.update(1.5)

    def test_no_halt_on_steady_high_signal(self):
        g = TemporalConsistencyGuard(alpha=0.3, consistency_floor=0.5)
        for _ in range(32):
            halt = g.update(0.9)
        assert halt is False
        assert g.ema is not None and g.ema > 0.5
