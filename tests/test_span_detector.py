# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — token-level span detector tests

from __future__ import annotations

import pytest

from director_ai.core.scoring.span_detector import (
    DEFAULT_SPAN_MODEL,
    HallucinatedSpan,
    HallucinationSpanDetector,
    SpanDetection,
    merge_flagged_spans,
)


class TestMergeFlaggedSpans:
    def test_no_tokens_flagged_returns_empty(self) -> None:
        resp = "all grounded text here"
        offsets = [(0, 3), (4, 12), (13, 17)]
        scores = [0.1, 0.2, 0.05]
        spans, flagged, mx = merge_flagged_spans(offsets, scores, resp, 0.95)
        assert spans == []
        assert flagged == 0
        assert mx == pytest.approx(0.2)

    def test_single_flagged_token_one_span(self) -> None:
        resp = "Paris Berlin Rome"
        offsets = [(0, 5), (6, 12), (13, 17)]
        scores = [0.1, 0.99, 0.2]
        spans, flagged, mx = merge_flagged_spans(offsets, scores, resp, 0.95)
        assert flagged == 1
        assert len(spans) == 1
        assert spans[0].text == "Berlin"
        assert spans[0].start == 6 and spans[0].end == 12
        assert spans[0].score == pytest.approx(0.99)
        assert mx == pytest.approx(0.99)

    def test_contiguous_tokens_bridge_whitespace(self) -> None:
        # Two flagged tokens separated only by a space merge into one phrase.
        resp = "offers takeout seating"
        offsets = [(0, 6), (7, 14), (15, 22)]
        scores = [0.98, 0.97, 0.1]
        spans, flagged, _ = merge_flagged_spans(offsets, scores, resp, 0.95)
        assert flagged == 2
        assert len(spans) == 1
        assert spans[0].text == "offers takeout"
        assert spans[0].score == pytest.approx(0.98)

    def test_gap_with_grounded_token_splits_spans(self) -> None:
        # A non-whitespace grounded token between two flagged ones splits them.
        resp = "wrong ok wrong"
        offsets = [(0, 5), (6, 8), (9, 14)]
        scores = [0.99, 0.1, 0.99]
        spans, flagged, _ = merge_flagged_spans(offsets, scores, resp, 0.95)
        assert flagged == 2
        assert [s.text for s in spans] == ["wrong", "wrong"]

    def test_zero_width_offsets_skipped(self) -> None:
        # Special tokens carry (0, 0) offsets and must never become spans.
        resp = "text"
        offsets = [(0, 0), (0, 4)]
        scores = [0.99, 0.99]
        spans, flagged, _ = merge_flagged_spans(offsets, scores, resp, 0.95)
        assert flagged == 1
        assert spans[0].text == "text"

    def test_threshold_boundary_inclusive(self) -> None:
        resp = "edge"
        spans, flagged, _ = merge_flagged_spans([(0, 4)], [0.95], resp, 0.95)
        assert flagged == 1 and len(spans) == 1


class TestSpanDetection:
    def test_coverage_fraction(self) -> None:
        d = SpanDetection(True, (), 0.9, 3, 12)
        assert d.coverage == pytest.approx(0.25)

    def test_coverage_zero_when_no_tokens(self) -> None:
        d = SpanDetection(False, (), 0.0, 0, 0)
        assert d.coverage == 0.0


class TestDetectorConstruction:
    def test_requires_model_and_tokenizer(self) -> None:
        with pytest.raises(ValueError, match="required"):
            HallucinationSpanDetector(None, object())
        with pytest.raises(ValueError, match="required"):
            HallucinationSpanDetector(object(), None)

    def test_threshold_range_validated(self) -> None:
        with pytest.raises(ValueError, match="token_threshold"):
            HallucinationSpanDetector(object(), object(), token_threshold=1.5)

    def test_min_tokens_validated(self) -> None:
        with pytest.raises(ValueError, match="min_tokens"):
            HallucinationSpanDetector(object(), object(), min_tokens=0)

    def test_default_model_id(self) -> None:
        assert DEFAULT_SPAN_MODEL == "anulum/director-ragtruth-token-modernbert"


# ── detect() glue, exercised with a lightweight fake model + tokenizer ──

torch = pytest.importorskip("torch")


class _FakeEnc(dict):
    """Mimic a transformers BatchEncoding for the slice detect() touches."""

    def __init__(self, seq_ids, offsets):
        super().__init__(input_ids=torch.tensor([[0] * len(seq_ids)]))
        self["offset_mapping"] = torch.tensor([offsets])
        self._seq_ids = seq_ids

    def sequence_ids(self):
        return self._seq_ids


class _FakeTokenizer:
    def __init__(self, seq_ids, offsets):
        self._seq_ids = seq_ids
        self._offsets = offsets

    def __call__(self, *_a, **_k):
        return _FakeEnc(self._seq_ids, self._offsets)


class _FakeModel(torch.nn.Module):
    """Returns fixed per-token logits; index 1 is the hallucinated class."""

    def __init__(self, hallucinated_probs):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)  # gives .parameters() a device
        self._probs = hallucinated_probs

    def forward(self, **_inputs):
        rows = [[1.0 - p, p] for p in self._probs]
        logits = torch.log(torch.tensor([rows]).clamp_min(1e-6))

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


def _detector(seq_ids, offsets, probs, **kw):
    return HallucinationSpanDetector(
        _FakeModel(probs), _FakeTokenizer(seq_ids, offsets), **kw
    )


def test_detect_empty_response_short_circuits() -> None:
    det = _detector([None], [(0, 0)], [0.0])
    out = det.detect("ctx", "   ")
    assert out == SpanDetection(False, (), 0.0, 0, 0)


def test_detect_flags_response_span_only() -> None:
    # tokens: [CLS] ctx ctx [SEP] resp resp resp ; only response tokens scored.
    resp = "Paris is wrong"
    seq_ids = [None, 0, 0, None, 1, 1, 1]
    offsets = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 5), (6, 8), (9, 14)]
    probs = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.99]  # only "wrong" hallucinated
    out = _detector(seq_ids, offsets, probs).detect("any context", resp)
    assert out.hallucinated is True
    assert out.response_tokens == 3
    assert out.flagged_tokens == 1
    assert [s.text for s in out.spans] == ["wrong"]
    assert out.max_token_score == pytest.approx(0.99, abs=1e-4)


def test_detect_min_tokens_gate() -> None:
    resp = "one two"
    seq_ids = [1, 1]
    offsets = [(0, 3), (4, 7)]
    probs = [0.99, 0.1]
    # one flagged token, min_tokens=2 -> not flagged despite a span existing
    out = _detector(seq_ids, offsets, probs, min_tokens=2).detect("c", resp)
    assert out.hallucinated is False
    assert out.flagged_tokens == 1


def test_detect_all_grounded() -> None:
    resp = "fully supported answer"
    seq_ids = [1, 1, 1]
    offsets = [(0, 5), (6, 15), (16, 22)]
    out = _detector(seq_ids, offsets, [0.01, 0.02, 0.03]).detect("c", resp)
    assert out.hallucinated is False
    assert out.spans == ()
    assert out.coverage == 0.0


class TestGuardWiring:
    """``ProductionGuard.span_detector`` / ``detect_spans`` opt-in gate."""

    def _guard(self, **cfg_kw):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        return ProductionGuard(config=DirectorConfig(**cfg_kw))

    def test_disabled_raises(self) -> None:
        guard = self._guard(span_detection_enabled=False)
        with pytest.raises(RuntimeError, match="span detection is disabled"):
            _ = guard.span_detector

    def test_enabled_builds_and_delegates(self, monkeypatch) -> None:
        sentinel = _detector([1], [(0, 4)], [0.99])
        captured = {}

        def fake_from_pretrained(model_id, **kw):
            captured["model_id"] = model_id
            captured["kw"] = kw
            return sentinel

        monkeypatch.setattr(
            "director_ai.core.scoring.span_detector.HallucinationSpanDetector"
            ".from_pretrained",
            classmethod(lambda cls, model_id, **kw: fake_from_pretrained(model_id, **kw)),
        )
        guard = self._guard(
            span_detection_enabled=True,
            span_token_threshold=0.9,
            span_min_tokens=2,
        )
        assert guard.span_detector is sentinel
        assert captured["model_id"] == "anulum/director-ragtruth-token-modernbert"
        assert captured["kw"]["token_threshold"] == 0.9
        assert captured["kw"]["min_tokens"] == 2
        # second access is cached (no rebuild)
        assert guard.span_detector is sentinel
        # detect_spans delegates to the detector
        out = guard.detect_spans("ctx", "wrong")
        assert isinstance(out, SpanDetection)
