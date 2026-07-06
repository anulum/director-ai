# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — span detector real-surface tests
"""Public-surface coverage for token-level hallucinated-span detection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from director_ai.core.config import DirectorConfig
from director_ai.core.scoring.span_detector import HallucinationSpanDetector


class _TorchEncoding(dict[str, torch.Tensor]):
    """Minimal tokenizer output that follows the BatchEncoding public protocol."""

    def __init__(
        self,
        *,
        sequence_ids: tuple[int | None, ...],
        offsets: tuple[tuple[int, int], ...],
    ) -> None:
        super().__init__(
            input_ids=torch.zeros((1, len(sequence_ids)), dtype=torch.long),
            offset_mapping=torch.tensor([offsets], dtype=torch.long),
        )
        self._sequence_ids = sequence_ids

    def sequence_ids(self) -> list[int | None]:
        """Return tokenizer segment ids for context and response tokens."""
        return list(self._sequence_ids)


class _Tokenizer:
    """Tokenizer test double that records the production truncation contract."""

    def __init__(
        self,
        *,
        sequence_ids: tuple[int | None, ...],
        offsets: tuple[tuple[int, int], ...],
    ) -> None:
        self._sequence_ids = sequence_ids
        self._offsets = offsets
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        context: str,
        response: str,
        **kwargs: object,
    ) -> _TorchEncoding:
        """Return deterministic token offsets for the detector public call."""
        self.calls.append(
            {
                "context": context,
                "response": response,
                **kwargs,
            }
        )
        return _TorchEncoding(sequence_ids=self._sequence_ids, offsets=self._offsets)


@dataclass(frozen=True, slots=True)
class _ModelOutput:
    """Token-classifier output carrying logits."""

    logits: torch.Tensor


class _TokenClassifier(torch.nn.Module):
    """Tiny token classifier that emits caller-supplied logits."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._device_anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self._logits = logits

    def forward(self, **_inputs: torch.Tensor) -> _ModelOutput:
        """Return logits in the HuggingFace token-classification shape."""
        return _ModelOutput(logits=self._logits.to(self._device_anchor.device))


def test_detector_detect_uses_public_token_classifier_surface() -> None:
    """A real torch-backed detector should flag only response-token spans."""
    response = "Paris is wrong"
    tokenizer = _Tokenizer(
        sequence_ids=(None, 0, 0, None, 1, 1, 1),
        offsets=((0, 0), (0, 0), (0, 0), (0, 0), (0, 5), (6, 8), (9, 14)),
    )
    model = _TokenClassifier(
        torch.tensor(
            [
                [
                    (5.0, 0.0),
                    (5.0, 0.0),
                    (5.0, 0.0),
                    (5.0, 0.0),
                    (5.0, 0.0),
                    (5.0, 0.0),
                    (0.0, 6.0),
                ]
            ],
            dtype=torch.float32,
        )
    )
    detector = HallucinationSpanDetector(
        model,
        tokenizer,
        token_threshold=0.95,
        max_length=32,
    )

    detection = detector.detect("The source only says Paris is a city.", response)

    assert detection.hallucinated is True
    assert detection.response_tokens == 3
    assert detection.flagged_tokens == 1
    assert detection.spans[0].text == "wrong"
    assert tokenizer.calls[0]["truncation"] == "only_first"
    assert tokenizer.calls[0]["max_length"] == 32
    assert tokenizer.calls[0]["return_offsets_mapping"] is True
    assert tokenizer.calls[0]["return_tensors"] == "pt"


def test_span_detector_rejects_non_positive_max_length() -> None:
    """Detector and config construction should reject unusable token budgets."""
    model = _TokenClassifier(torch.zeros((1, 1, 2), dtype=torch.float32))
    tokenizer = _Tokenizer(sequence_ids=(1,), offsets=((0, 4),))

    with pytest.raises(ValueError, match="max_length"):
        HallucinationSpanDetector(model, tokenizer, max_length=0)

    with pytest.raises(ValueError, match="span_max_length"):
        DirectorConfig(span_max_length=0)

    with pytest.raises(ValueError, match="span_token_threshold"):
        DirectorConfig(span_token_threshold=-0.1)

    with pytest.raises(ValueError, match="span_min_tokens"):
        DirectorConfig(span_min_tokens=0)


def test_general_mode_preserves_requested_nli_and_disables_retrieval() -> None:
    """General mode should preserve explicit NLI and disable retrieval layers."""
    config = DirectorConfig(
        mode="general",
        use_nli=True,
        hybrid_retrieval=True,
        reranker_enabled=True,
    )

    assert config.use_nli is True
    assert config.hybrid_retrieval is False
    assert config.reranker_enabled is False


def test_production_config_rejects_model_backed_coherence_without_nli() -> None:
    """Production mode should fail closed when coherence NLI is disabled."""
    with pytest.raises(ValueError, match="coherence_require_model_backed_nli"):
        DirectorConfig(
            mode="auto",
            production_mode=True,
            use_nli=False,
            hybrid_retrieval=False,
            reranker_enabled=False,
            coherence_require_model_backed_nli=True,
            api_keys=["prod-key"],
            llm_provider="openai",
            knowledge_write_hmac_keys=(
                '{"kid-1":"signing-secret-at-least-32-chars-xx"}'
            ),
        )
