# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction scorer tests (model-mocked, offline)

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from director_ai.core.scoring.contradiction import (  # noqa: E402
    ContradictionResult,
    ContradictionScorer,
)


class _StubConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _PassThrough:
    """Carries the hypotheses through the ``.to(device)`` call the scorer makes."""

    def __init__(self, hyps):
        self.hyps = list(hyps)

    def to(self, _device):
        return self


class _StubModel:
    """Returns fixed per-pair logits; contradiction is index 2."""

    def __init__(self, logits_by_hyp, id2label=None):
        self._logits_by_hyp = logits_by_hyp
        self.config = _StubConfig(
            id2label or {0: "entailment", 1: "neutral", 2: "contradiction"}
        )
        self.device = "cpu"

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, **enc):
        rows = [self._logits_by_hyp[h] for h in enc["_hyps"].hyps]

        class _Out:
            logits = torch.tensor(rows, dtype=torch.float32)

        return _Out()


class _StubTokenizer:
    """Passes the raw hypotheses through so the stub model can look them up."""

    def __call__(self, premises, hypotheses, **kw):
        return {"_hyps": _PassThrough(hypotheses)}


def _scorer(logits_by_hyp, *, threshold=0.2, id2label=None):
    model = _StubModel(logits_by_hyp, id2label=id2label)
    return ContradictionScorer(
        model, _StubTokenizer(), contradiction_idx=2, threshold=threshold
    )


def test_requires_model_and_tokenizer():
    with pytest.raises(ValueError, match="required"):
        ContradictionScorer(None, object(), contradiction_idx=2)


def test_contradiction_probability_is_softmax_of_class():
    # logits favouring contradiction (index 2)
    sc = _scorer({"bad claim": [0.0, 0.0, 5.0]})
    p = sc.contradiction("the fact", "bad claim")
    assert p == pytest.approx(torch.softmax(torch.tensor([0.0, 0.0, 5.0]), 0)[2].item())
    assert p > 0.9


def test_entailed_claim_low_contradiction():
    sc = _scorer({"good claim": [5.0, 0.0, 0.0]})  # entailment dominates
    assert sc.contradiction("the fact", "good claim") < 0.05


def test_neutral_claim_low_contradiction():
    # A correct-but-unsupported claim is NEUTRAL, not contradiction -> must NOT halt.
    sc = _scorer({"unsupported": [0.0, 5.0, 0.0]})
    assert sc.contradiction("the fact", "unsupported") < 0.05
    assert sc.contradicts("the fact", "unsupported") is False


def test_contradicts_threshold():
    sc = _scorer({"x": [0.0, 0.0, 1.0]}, threshold=0.5)
    p = sc.contradiction("f", "x")
    assert sc.contradicts("f", "x") is (p >= 0.5)


def test_judge_returns_result():
    sc = _scorer({"x": [0.0, 0.0, 5.0]}, threshold=0.2)
    r = sc.judge("f", "x")
    assert isinstance(r, ContradictionResult)
    assert r.contradicts is True
    assert r.contradiction > 0.9


def test_empty_text_is_zero_without_inference():
    sc = _scorer({})
    assert sc.contradiction("", "x") == 0.0
    assert sc.contradiction("f", "   ") == 0.0
    assert sc.contradiction("f", None) == 0.0  # type: ignore[arg-type]


def test_batch_matches_single():
    sc = _scorer({"a": [0.0, 0.0, 5.0], "b": [5.0, 0.0, 0.0]})
    batch = sc.contradiction_batch([("f", "a"), ("f", "b")])
    assert batch[0] > 0.9
    assert batch[1] < 0.05
    assert sc.contradiction_batch([]) == []


def test_resolve_contradiction_idx_variants():
    assert (
        ContradictionScorer._resolve_contradiction_idx(
            _StubModel({}, id2label={0: "entail", 1: "neutral", 2: "CONTRADICTION"})
        )
        == 2
    )
    # 'refuted' style label also recognised
    assert (
        ContradictionScorer._resolve_contradiction_idx(
            _StubModel({}, id2label={0: "SUPPORTED", 1: "refuted"})
        )
        == 1
    )
    # two-class supported/not-supported has no contradiction class
    assert (
        ContradictionScorer._resolve_contradiction_idx(
            _StubModel({}, id2label={0: "LABEL_0", 1: "LABEL_1"})
        )
        is None
    )
    # a model without an id2label map resolves to None
    no_labels = type("M", (), {"config": type("C", (), {"id2label": None})()})()
    assert ContradictionScorer._resolve_contradiction_idx(no_labels) is None


def test_from_pretrained_without_transformers(monkeypatch):
    import builtins

    real = builtins.__import__

    def _no_tf(name, *a, **k):
        if name == "transformers":
            raise ImportError("no transformers")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_tf)
    with pytest.raises(ImportError, match="requires transformers"):
        ContradictionScorer.from_pretrained()


def test_threshold_property():
    sc = _scorer({}, threshold=0.37)
    assert sc.threshold == pytest.approx(0.37)


def test_from_pretrained_loads_and_resolves_contradiction_index(monkeypatch):
    import transformers

    model = _StubModel({}, id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: _StubTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: model,
    )
    sc = ContradictionScorer.from_pretrained(
        "some/model", revision="0" * 40, threshold=0.3
    )
    assert sc.threshold == pytest.approx(0.3)
    assert sc._ci == 2


def test_from_pretrained_rejects_model_without_contradiction_class(monkeypatch):
    import transformers

    two_class = _StubModel({}, id2label={0: "supported", 1: "not_supported"})
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: _StubTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: two_class,
    )
    with pytest.raises(ValueError, match="no contradiction class"):
        ContradictionScorer.from_pretrained("two/class", revision="0" * 40)


def test_from_pretrained_moves_model_to_requested_cuda_device(monkeypatch):
    import torch
    import transformers

    model = _StubModel({}, id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    moved: dict[str, str] = {}
    model.to = lambda device: moved.setdefault("device", device) or model
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: model,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    sc = ContradictionScorer.from_pretrained("m/x", revision="0" * 40, device=0)
    assert moved["device"] == "cuda:0"
    assert sc._ci == 2
