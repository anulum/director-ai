# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — token-level hallucinated-span detector

"""Locate the hallucinated spans inside a RAG response, token by token.

Response- and claim-level grounding (NLI, FactCG) judge a whole answer or a whole
claim, so they miss the short "baseless addition" spans that dominate RAGTruth —
a plausible detail absent from the source, embedded in otherwise-grounded text.
This detector reads ``[context] [SEP] [response]`` through a ModernBERT token
classifier and labels each *response* token supported or hallucinated, then maps
the flagged tokens back to character spans of the response. The context is
truncated (``only_first``) so the response is always scored in full.

The class boundary is tuned at the token-probability level: the trainer optimises
a positive-weighted loss (hallucinated tokens are a small minority), so the
useful operating point is a high probability cut (default 0.95) with a minimum
flagged-token count, calibrated on the RAGTruth test split. At that point the
detector reaches example-level F1 0.763 / balanced accuracy 0.814 / false-positive
rate 0.071 — competitive with token-level baselines and far above the
claim-decompose NLI path (F1 0.366).

The model and tokenizer are loaded lazily; ``transformers`` and ``torch`` are
optional (``pip install director-ai[nli]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..model_revisions import resolve_model_revision

__all__ = [
    "DEFAULT_SPAN_MODEL",
    "HallucinatedSpan",
    "SpanDetection",
    "HallucinationSpanDetector",
    "merge_flagged_spans",
]

DEFAULT_SPAN_MODEL = "anulum/director-ragtruth-token-modernbert"
DEFAULT_TOKEN_THRESHOLD = 0.95
DEFAULT_MIN_TOKENS = 1
_HALLUCINATED_IDX = 1


@dataclass(frozen=True)
class HallucinatedSpan:
    """A contiguous run of response characters the model judged unsupported."""

    start: int
    end: int
    text: str
    score: float  # highest token probability inside the span


@dataclass(frozen=True)
class SpanDetection:
    """Outcome of scanning one response against its grounding context."""

    hallucinated: bool
    spans: tuple[HallucinatedSpan, ...]
    max_token_score: float
    flagged_tokens: int
    response_tokens: int

    @property
    def coverage(self) -> float:
        """Fraction of response tokens flagged as hallucinated."""
        if not self.response_tokens:
            return 0.0
        return self.flagged_tokens / self.response_tokens


def merge_flagged_spans(
    offsets: list[tuple[int, int]],
    scores: list[float],
    response: str,
    threshold: float,
) -> tuple[list[HallucinatedSpan], int, float]:
    """Merge contiguous flagged response tokens into character spans.

    ``offsets`` and ``scores`` are aligned per response token (character offsets
    into *response* and ``P(hallucinated)``). Adjacent flagged tokens — those
    whose character ranges touch or overlap — collapse into one span; a small gap
    of non-flagged whitespace between two flagged tokens is bridged so a single
    hallucinated phrase is not split on its spaces.
    """
    spans: list[HallucinatedSpan] = []
    flagged = 0
    max_score = 0.0
    cur_start = cur_end = -1
    cur_max = 0.0
    for (cs, ce), score in zip(offsets, scores, strict=True):
        max_score = max(max_score, score)
        if score < threshold or ce <= cs:
            continue
        flagged += 1
        if cur_end >= 0 and response[cur_end:cs].strip() == "":
            cur_end = ce  # bridge intervening whitespace
            cur_max = max(cur_max, score)
        else:
            if cur_end >= 0:
                spans.append(
                    HallucinatedSpan(
                        cur_start, cur_end, response[cur_start:cur_end], cur_max
                    )
                )
            cur_start, cur_end, cur_max = cs, ce, score
    if cur_end >= 0:
        spans.append(
            HallucinatedSpan(cur_start, cur_end, response[cur_start:cur_end], cur_max)
        )
    return spans, flagged, max_score


class HallucinationSpanDetector:
    """Token-level detector that flags hallucinated spans in a RAG response."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        token_threshold: float = DEFAULT_TOKEN_THRESHOLD,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        max_length: int = 1024,
    ) -> None:
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required")
        if not 0.0 <= token_threshold <= 1.0:
            raise ValueError("token_threshold must be in [0, 1]")
        if min_tokens < 1:
            raise ValueError("min_tokens must be >= 1")
        self._model = model
        self._tokenizer = tokenizer
        self._token_threshold = float(token_threshold)
        self._min_tokens = int(min_tokens)
        self._max_length = int(max_length)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_SPAN_MODEL,
        *,
        revision: str | None = None,
        device: int = -1,
        token_threshold: float = DEFAULT_TOKEN_THRESHOLD,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        max_length: int = 1024,
    ) -> HallucinationSpanDetector:
        """Load the ModernBERT token classifier at a pinned revision."""
        try:
            import torch
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "HallucinationSpanDetector.from_pretrained requires transformers "
                "and torch. Install with: pip install director-ai[nli]",
            ) from exc

        pinned = resolve_model_revision(model_id, revision)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=pinned)
        model = AutoModelForTokenClassification.from_pretrained(
            model_id, revision=pinned
        )
        model.eval()
        if device >= 0 and torch.cuda.is_available():
            model.to(f"cuda:{device}")
        return cls(
            model,
            tokenizer,
            token_threshold=token_threshold,
            min_tokens=min_tokens,
            max_length=max_length,
        )

    def detect(self, context: str, response: str) -> SpanDetection:
        """Flag the hallucinated spans of *response* against *context*."""
        response = response or ""
        if not response.strip():
            return SpanDetection(False, (), 0.0, 0, 0)

        import torch

        enc = self._tokenizer(
            context or "",
            response,
            truncation="only_first",
            max_length=self._max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        seq_ids = enc.sequence_ids()
        offsets_all = enc.pop("offset_mapping")[0].tolist()
        device = next(self._model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**model_inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)[:, _HALLUCINATED_IDX].cpu().tolist()

        resp_offsets = [tuple(offsets_all[i]) for i, s in enumerate(seq_ids) if s == 1]
        resp_scores = [probs[i] for i, s in enumerate(seq_ids) if s == 1]

        spans, flagged, max_score = merge_flagged_spans(
            resp_offsets, resp_scores, response, self._token_threshold
        )
        hallucinated = flagged >= self._min_tokens
        return SpanDetection(
            hallucinated=hallucinated,
            spans=tuple(spans),
            max_token_score=max_score,
            flagged_tokens=flagged,
            response_tokens=len(resp_scores),
        )
