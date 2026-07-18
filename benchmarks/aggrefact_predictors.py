# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AggreFact Predictors

"""Model predictors for the AggreFact benchmark: the FactCG instruction
template with SummaC-style source chunking, the binary NLI predictor,
and the NLIScorer bidirectional-chunking wrapper.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("DirectorAI.Benchmark.AggreFact")

_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def _normalise_scorer_template(template: str | None) -> str:
    value = (template or os.environ.get("DIRECTOR_SCORER_TEMPLATE", "auto")).strip()
    if not value:
        return "auto"
    allowed = {"auto", "factcg", "sequence-pair"}
    if value not in allowed:
        raise ValueError(
            f"scorer template must be one of auto, factcg, sequence-pair; got {value!r}"
        )
    return value


def _uses_factcg_template(
    model_name: str,
    model_config: object,
    template: str | None,
) -> bool:
    mode = _normalise_scorer_template(template)
    if mode == "factcg":
        return True
    if mode == "sequence-pair":
        return False
    return "factcg" in model_name.lower() or bool(
        getattr(model_config, "factcg", False)
    )


def _chunk_source(text: str, max_tokens: int = 550) -> list[str]:
    """Split source document into sentence-level chunks (SummaC-style)."""
    import nltk

    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sents = nltk.sent_tokenize(text)

    chunks: list[str] = []
    chunk, chunk_len = "", 0
    for s in sents:
        s_len = len(s.split())
        if chunk and chunk_len + s_len > max_tokens:
            chunks.append(chunk)
            chunk, chunk_len = s, s_len
        else:
            chunk = f"{chunk}\n{s}".strip("\n") if chunk else s
            chunk_len += s_len
    if chunk:
        chunks.append(chunk)
    return chunks or [text]


class _BinaryNLIPredictor:
    """NLI model wrapped for binary factual consistency scoring.

    Returns entailment probability as the "supported" score.
    FactCG models use instruction template + SummaC-style source chunking.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int = 2048,
        scorer_template: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name or os.environ.get(
            "DIRECTOR_NLI_MODEL",
            "yaxili96/FactCG-DeBERTa-v3-Large",
        )
        logger.info("Loading NLI model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self._torch = torch
        self._num_labels = self.model.config.num_labels
        self._is_factcg = _uses_factcg_template(
            self.model_name,
            self.model.config,
            scorer_template,
        )
        logger.info(
            "Model loaded on %s (%s, %d labels, factcg=%s)",
            self.device,
            self.model_name,
            self._num_labels,
            self._is_factcg,
        )

    def _score_single(self, premise: str, hypothesis: str) -> float:
        """Score a single (premise, hypothesis) pair."""
        if self._is_factcg:
            text = _FACTCG_TEMPLATE.format(text_a=premise, text_b=hypothesis)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        else:
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.softmax(logits, dim=1).cpu().numpy()[0]
        if self._num_labels == 2:
            return float(probs[1])
        return float(probs[0])

    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batched forward pass — all pairs in one call."""
        if self._is_factcg:
            texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
        else:
            premises = [p for p, _ in pairs]
            hypotheses = [h for _, h in pairs]
            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.softmax(logits, dim=1).cpu().numpy()
        if self._num_labels == 2:
            return [float(row[1]) for row in probs]
        return [float(row[0]) for row in probs]

    def score(self, premise: str, hypothesis: str) -> float:
        """Return P(supported) with SummaC source chunking for FactCG.

        Splits premise into sentence chunks, scores each vs hypothesis,
        returns max (matching official FactCG evaluation).
        Chunks are batched into a single forward pass.
        """
        if not self._is_factcg:
            return self._score_single(premise, hypothesis)
        chunks = _chunk_source(premise)
        if len(chunks) == 1:
            return self._score_single(chunks[0], hypothesis)
        try:
            return max(self._score_batch([(c, hypothesis) for c in chunks]))
        except (RuntimeError, self._torch.OutOfMemoryError):
            # Fall back to per-chunk scoring on OOM (8 GB GPUs hit this on
            # long premises with many chunks).
            self._torch.cuda.empty_cache()
            return max(self._score_single(c, hypothesis) for c in chunks)


class _NLIScorerPredictor:
    """Wraps NLIScorer.score_chunked() for bidirectional chunking comparison."""

    def __init__(self, model_name: str | None = None, overlap_ratio: float = 0.0):
        import torch

        from director_ai.core.nli import NLIScorer

        self.scorer = NLIScorer(
            use_model=True,
            model_name=model_name
            or os.environ.get("DIRECTOR_NLI_MODEL", "yaxili96/FactCG-DeBERTa-v3-Large"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self._overlap_ratio = overlap_ratio
        logger.info(
            "NLIScorerPredictor ready (overlap=%.2f)",
            overlap_ratio,
        )

    def score(self, premise: str, hypothesis: str) -> float:
        score, _ = self.scorer.score_chunked(
            premise,
            hypothesis,
            overlap_ratio=self._overlap_ratio,
        )
        return 1.0 - score
