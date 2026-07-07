# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — token-level span detector tests

from __future__ import annotations

import random
import sys
from types import SimpleNamespace

import pytest

import director_ai.core.scoring.span_detector as span_module
from director_ai.core.scoring.span_detector import (
    DEFAULT_SPAN_MODEL,
    HallucinationSpanDetector,
    SpanDetection,
    _merge_flagged_spans_py,
    merge_flagged_spans,
)

try:
    import torch

    _USING_FAKE_TORCH = False
except ImportError:  # pragma: no cover - environment-dependent optional extra
    _USING_FAKE_TORCH = True

    class _FakeInputTensor:
        def to(self, _device: str) -> _FakeInputTensor:
            return self

    class _FakeOffsetTensor:
        def __init__(self, offsets):
            self._offsets = offsets

        def __getitem__(self, index: int) -> _FakeOffsetTensor:
            assert index == 0
            return self

        def tolist(self):
            return self._offsets

    class _FakeProbabilityVector:
        def __init__(self, probs):
            self._probs = probs

        def cpu(self) -> _FakeProbabilityVector:
            return self

        def tolist(self):
            return self._probs

    class _FakeProbabilityMatrix:
        def __init__(self, probs):
            self._probs = probs

        def __getitem__(self, key):
            rows, label_idx = key
            assert rows == slice(None)
            assert label_idx == 1
            return _FakeProbabilityVector(self._probs)

    class _FakeLogits:
        def __init__(self, probs):
            self.probs = probs

        def __getitem__(self, index: int) -> _FakeLogits:
            assert index == 0
            return self

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            return False

    torch = SimpleNamespace(  # type: ignore[assignment]
        FakeLogits=_FakeLogits,
        cuda=SimpleNamespace(is_available=lambda: False),
        nn=SimpleNamespace(Module=object, Linear=lambda *_args, **_kwargs: object()),
        no_grad=lambda: _NoGrad(),
        softmax=lambda logits, *, dim: _FakeProbabilityMatrix(logits.probs),
        tensor=lambda _value: _FakeInputTensor(),
    )

@pytest.fixture(autouse=True)
def _scope_fake_torch():
    """Expose the fake ``torch`` for this module's tests only.

    The span detector imports ``torch`` lazily inside its functions, so the fake
    must live in ``sys.modules`` while a test runs. Injecting it at import time
    (as this module previously did) leaked the fake into the whole pytest
    session, so any later test guarding on ``pytest.importorskip("torch")``
    received the fake instead of skipping. Scope it per test, with cleanup.
    """
    if not _USING_FAKE_TORCH:
        yield
        return
    original = sys.modules.get("torch")
    sys.modules["torch"] = torch
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


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


class TestRustPythonParity:
    """The Rust accelerator and the Python floor must agree bit-for-bit."""

    def test_python_floor_matches_default_dispatch(self) -> None:
        # The floor produces the same spans/flagged/max as the public dispatcher.
        resp = "wrong ok wrong phrase here"
        offsets = [(0, 5), (6, 8), (9, 14), (15, 21), (22, 26)]
        scores = [0.99, 0.1, 0.97, 0.96, 0.2]
        assert merge_flagged_spans(
            offsets, scores, resp, 0.95
        ) == _merge_flagged_spans_py(offsets, scores, resp, 0.95)

    def test_fallback_to_python_floor_when_rust_absent(self, monkeypatch) -> None:
        # Force the Python-floor branch of the dispatcher and confirm it is taken.
        monkeypatch.setattr(span_module, "_RUST_SPAN_MERGE", False)
        resp = "Paris is wrong here"
        offsets = [(0, 5), (6, 8), (9, 14), (15, 19)]
        scores = [0.1, 0.2, 0.99, 0.96]
        spans, flagged, mx = merge_flagged_spans(offsets, scores, resp, 0.95)
        assert flagged == 2
        assert [s.text for s in spans] == ["wrong here"]
        assert mx == pytest.approx(0.99)

    def test_differential_sweep_rust_vs_floor(self) -> None:
        # A seeded random sweep over varied offsets, thresholds, and a charset that
        # includes Unicode and C0 separators (the str.isspace edge) finds no drift.
        if not span_module._RUST_SPAN_MERGE:  # pragma: no cover - accelerator absent
            pytest.skip("Rust span-merge accelerator not installed")
        rng = random.Random(20260616)
        charset = "ab cd.ef\n\t\x1f  xyz,"
        for _ in range(3000):
            resp = "".join(rng.choice(charset) for _ in range(rng.randint(0, 50)))
            offsets, scores, pos = [], [], 0
            while pos < len(resp) and len(offsets) < 30:
                cs = pos
                ce = min(len(resp), pos + rng.randint(0, 3))
                offsets.append((cs, ce))
                scores.append(round(rng.random(), 3))
                pos = ce + rng.randint(0, 2)
            threshold = rng.choice([0.0, 0.5, 0.95, 1.0])
            assert merge_flagged_spans(offsets, scores, resp, threshold) == (
                _merge_flagged_spans_py(offsets, scores, resp, threshold)
            ), (offsets, scores, repr(resp), threshold)


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

    def test_from_pretrained_uses_pinned_revision_and_cuda_device(
        self, monkeypatch
    ) -> None:
        calls: list[tuple[str, str, str]] = []

        class FakeModel:
            def __init__(self) -> None:
                self.eval_called = False
                self.device = ""

            def eval(self) -> None:
                self.eval_called = True

            def to(self, device: str) -> None:
                self.device = device

        fake_model = FakeModel()
        fake_tokenizer = object()

        class FakeTokenizerLoader:
            @staticmethod
            def from_pretrained(model_id: str, *, revision: str) -> object:
                calls.append(("tokenizer", model_id, revision))
                return fake_tokenizer

        class FakeModelLoader:
            @staticmethod
            def from_pretrained(model_id: str, *, revision: str) -> FakeModel:
                calls.append(("model", model_id, revision))
                return fake_model

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoTokenizer=FakeTokenizerLoader,
                AutoModelForTokenClassification=FakeModelLoader,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        )
        monkeypatch.setattr(
            span_module,
            "resolve_model_revision",
            lambda model_id, revision: f"pinned:{model_id}:{revision}",
        )

        detector = HallucinationSpanDetector.from_pretrained(
            "model-id",
            revision="rev-a",
            device=2,
            token_threshold=0.7,
            min_tokens=3,
            max_length=128,
        )

        assert detector._model is fake_model
        assert detector._tokenizer is fake_tokenizer
        assert detector._token_threshold == pytest.approx(0.7)
        assert detector._min_tokens == 3
        assert detector._max_length == 128
        assert fake_model.eval_called is True
        assert fake_model.device == "cuda:2"
        assert calls == [
            ("tokenizer", "model-id", "pinned:model-id:rev-a"),
            ("model", "model-id", "pinned:model-id:rev-a"),
        ]

    def test_from_pretrained_default_device_does_not_move_model(
        self, monkeypatch
    ) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.device = ""

            def eval(self) -> None:
                pass

            def to(self, device: str) -> None:
                self.device = device

        fake_model = FakeModel()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=lambda *_args, **_kwargs: object()
                ),
                AutoModelForTokenClassification=SimpleNamespace(
                    from_pretrained=lambda *_args, **_kwargs: fake_model
                ),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        )
        monkeypatch.setattr(
            span_module,
            "resolve_model_revision",
            lambda _model_id, revision: revision or "rev",
        )

        detector = HallucinationSpanDetector.from_pretrained("model-id")

        assert detector._model is fake_model
        assert fake_model.device == ""


# ── detect() glue, exercised with a lightweight fake model + tokenizer ──


class _FakeEnc(dict):
    """Mimic a transformers BatchEncoding for the slice detect() touches."""

    def __init__(self, seq_ids, offsets):
        super().__init__(input_ids=torch.tensor([[0] * len(seq_ids)]))
        self["offset_mapping"] = (
            _FakeOffsetTensor(offsets) if _USING_FAKE_TORCH else torch.tensor([offsets])
        )
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

    def parameters(self):
        if _USING_FAKE_TORCH:
            yield SimpleNamespace(device="cpu")
            return
        yield from super().parameters()

    def __call__(self, **inputs):
        if _USING_FAKE_TORCH:
            return self.forward(**inputs)
        return super().__call__(**inputs)

    def forward(self, **_inputs):
        if _USING_FAKE_TORCH:
            logits = torch.FakeLogits(self._probs)
        else:
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
            classmethod(
                lambda cls, model_id, **kw: fake_from_pretrained(model_id, **kw)
            ),
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
