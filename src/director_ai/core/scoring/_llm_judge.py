# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LLM-as-Judge Escalation

"""LLM-as-judge subsystem for borderline NLI score escalation.

Extracted from scorer.py to reduce module size.  Used by
:class:`CoherenceScorer` via composition.
"""

from __future__ import annotations

import logging
import time

from ..metrics import metrics

# LLM-as-judge blending constants
LLM_JUDGE_AGREE_DIVERGENCE = 0.2
LLM_JUDGE_DISAGREE_DIVERGENCE = 0.8
LLM_JUDGE_LLM_WEIGHT = 0.3  # nli_w = 1.0 - llm_w * judge_conf

logger = logging.getLogger("DirectorAI")


class LLMJudge:
    """LLM-as-judge escalation engine.

    Supports three providers:

    * ``"local"`` — local DeBERTa-base binary classifier (no API calls).
    * ``"openai"`` — OpenAI Chat Completions API.
    * ``"anthropic"`` — Anthropic Messages API.

    The judge is consulted when the NLI softmax margin is below
    *confidence_threshold* (i.e. the model is uncertain).  The judge
    verdict is blended with the NLI score at a 30/70 ratio scaled by
    judge confidence.

    Parameters
    ----------
    provider : str
        ``"openai"``, ``"anthropic"``, or ``"local"``.
    model : str
        Model ID (HuggingFace path for local, API model name otherwise).
    confidence_threshold : float
        NLI margin below which to escalate (default 0.3).
    device : str | None
        Torch device for local judge.
    privacy_mode : bool
        Redact PII before sending to external judge.
    task_judge_thresholds : dict[str, float] | None
        Per-task-type escalation thresholds.
    """

    _DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
    }
    _JUDGE_CACHE_MAX = 256
    _JUDGE_RETRY_MAX = 3
    _JUDGE_RETRY_BACKOFF = (0.5, 1.0)

    def __init__(
        self,
        provider: str = "",
        model: str = "",
        confidence_threshold: float = 0.3,
        device: str | None = None,
        privacy_mode: bool = False,
        task_judge_thresholds: dict[str, float] | None = None,
        cost_callback=None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._judge_cache: dict[int, float] = {}
        self._privacy_mode = privacy_mode
        self._cost_callback = cost_callback

        # Local DeBERTa-base judge model
        self._local_judge_model = None
        self._local_judge_tokenizer = None
        self._local_judge_device = "cpu"

        # Per-task-type escalation thresholds
        self.task_judge_thresholds: dict[str, float] = task_judge_thresholds or {
            "dialogue": 0.35,
            "summarization": 0.25,
            "qa": 0.30,
            "fact_check": 0.20,
            "default": confidence_threshold,
        }

        if provider == "local" and model:
            self._init_local_judge(model, device)

    # -- Local judge initialisation ----------------------------------------

    def _init_local_judge(
        self,
        model_path: str,
        device: str | None = None,
    ) -> None:  # pragma: no cover
        """Load local DeBERTa-base judge model for borderline escalation."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._local_judge_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
            )
            self._local_judge_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=False,
                )
            )
            from .._device import select_torch_device

            self._local_judge_device = select_torch_device(device)
            assert self._local_judge_model is not None
            self._local_judge_model.to(self._local_judge_device)
            self._local_judge_model.eval()
            logger.info(
                "Local judge loaded: %s on %s",
                model_path,
                self._local_judge_device,
            )
        except Exception as exc:
            logger.warning("Failed to load local judge model: %s", exc)
            self._local_judge_model = None
            self._local_judge_tokenizer = None

    # -- Escalation decision -----------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when a judge provider is configured and usable."""
        if not self.provider:
            return False
        return not (self.provider == "local" and self._local_judge_model is None)

    def should_escalate(self, nli_score: float, task_type: str = "default") -> bool:
        """True when the judge should be consulted for *nli_score*."""
        if not self.enabled:
            return False
        threshold = self.task_judge_thresholds.get(
            task_type,
            self.confidence_threshold,
        )
        return bool(abs(nli_score - 0.5) < threshold)

    # -- Check dispatch ----------------------------------------------------

    def check(
        self,
        prompt: str,
        response: str,
        nli_score: float,
        *,
        redactor=None,
    ) -> float:
        """Escalate to judge and return adjusted divergence score.

        Routes to local DeBERTa judge or external LLM API depending on
        provider.  Returns raw *nli_score* on failure.
        """
        if self.provider == "local":
            return self._local_judge_check(prompt, response, nli_score)
        return self._llm_judge_check(prompt, response, nli_score, redactor=redactor)

    # -- Local judge path --------------------------------------------------

    def _local_judge_check(
        self,
        prompt: str,
        response: str,
        nli_score: float,
    ) -> float:
        """Local DeBERTa-base binary judge.

        Returns adjusted divergence via 70/30 blending.
        Falls back to raw nli_score if model unavailable.
        """
        if self._local_judge_model is None or self._local_judge_tokenizer is None:
            return nli_score
        return self._local_judge_infer(prompt, response, nli_score)

    def _local_judge_infer(  # pragma: no cover -- requires torch
        self,
        prompt: str,
        response: str,
        nli_score: float,
    ) -> float:
        """Run local judge forward pass and blend with NLI score."""
        import torch

        cache_key = hash((prompt[:500], response[:500]))
        cached = self._judge_cache.get(cache_key)
        if cached is not None:
            metrics.inc("llm_judge_cache_hits")
            return cached
        metrics.inc("llm_judge_cache_misses")

        t0 = time.monotonic()
        metrics.inc("llm_judge_escalations")

        judge_input = (
            f"NLI divergence: {nli_score:.2f}\n"
            f"Context: {prompt[:400]}\n"
            f"Response: {response[:400]}"
        )
        tokenizer = self._local_judge_tokenizer
        model = self._local_judge_model
        assert tokenizer is not None and model is not None
        inputs = tokenizer(
            judge_input,
            return_tensors="pt",
            max_length=384,
            truncation=True,
        )
        inputs = {k: v.to(self._local_judge_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        judge_agrees = probs[0] > 0.5  # class 0 = approve
        judge_conf = float(max(probs))
        llm_divergence = (
            LLM_JUDGE_AGREE_DIVERGENCE
            if judge_agrees
            else LLM_JUDGE_DISAGREE_DIVERGENCE
        )
        llm_w = LLM_JUDGE_LLM_WEIGHT * judge_conf
        nli_w = 1.0 - llm_w
        adjusted = max(0.0, min(1.0, nli_w * nli_score + llm_w * llm_divergence))

        if len(self._judge_cache) >= self._JUDGE_CACHE_MAX:
            self._judge_cache.pop(next(iter(self._judge_cache)))
        self._judge_cache[cache_key] = adjusted

        metrics.observe("llm_judge_seconds", time.monotonic() - t0)
        return adjusted

    # -- External LLM judge path -------------------------------------------

    def _llm_judge_check(
        self,
        prompt: str,
        response: str,
        nli_score: float,
        *,
        redactor=None,
    ) -> float:
        """Escalate to external LLM judge (OpenAI / Anthropic)."""
        import os

        cache_key = hash((prompt[:500], response[:500]))
        cached = self._judge_cache.get(cache_key)
        if cached is not None:
            metrics.inc("llm_judge_cache_hits")
            return cached
        metrics.inc("llm_judge_cache_misses")

        t0 = time.monotonic()
        metrics.inc("llm_judge_escalations")

        model = (
            os.environ.get("DIRECTOR_LLM_JUDGE_MODEL")
            or self.model
            or self._DEFAULT_MODELS.get(self.provider, "")
        )

        p_text = prompt[:500]
        r_text = response[:500]
        if self._privacy_mode and redactor is not None:
            p_text = redactor(p_text)
            r_text = redactor(r_text)

        judge_prompt = (
            f"Given the prompt: {p_text}\n"
            f"Response: {r_text}\n"
            f"NLI divergence score: {nli_score:.3f}\n"
            "Is this response factually correct? "
            'Reply with JSON: {"verdict": "YES" or "NO", "confidence": 0-100}'
        )
        try:
            reply = self._call_llm_judge(model, judge_prompt, nli_score)
            if reply is None:
                return nli_score

            llm_agrees, judge_conf = self._parse_judge_reply(reply)
            llm_divergence = (
                LLM_JUDGE_AGREE_DIVERGENCE
                if llm_agrees
                else LLM_JUDGE_DISAGREE_DIVERGENCE
            )
            llm_w = LLM_JUDGE_LLM_WEIGHT * judge_conf
            nli_w = 1.0 - llm_w
            adjusted = max(
                0.0,
                min(1.0, nli_w * nli_score + llm_w * llm_divergence),
            )
            if len(self._judge_cache) >= self._JUDGE_CACHE_MAX:
                self._judge_cache.pop(next(iter(self._judge_cache)))
            self._judge_cache[cache_key] = adjusted
            return adjusted
        finally:
            metrics.observe("llm_judge_seconds", time.monotonic() - t0)

    def _call_llm_judge(
        self,
        model: str,
        judge_prompt: str,
        fallback: float,
    ) -> str | None:
        """Call LLM provider with retry on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(self._JUDGE_RETRY_MAX):
            try:
                if self.provider == "openai":
                    import openai

                    openai_client = openai.OpenAI()
                    openai_result = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": judge_prompt}],
                        max_tokens=50,
                        response_format={"type": "json_object"},
                    )
                    if self._cost_callback and openai_result.usage:
                        self._cost_callback(
                            model,
                            openai_result.usage.prompt_tokens,
                            openai_result.usage.completion_tokens,
                        )
                    return openai_result.choices[0].message.content or ""
                if self.provider == "anthropic":
                    import anthropic

                    anthropic_client = anthropic.Anthropic()
                    anthropic_result = anthropic_client.messages.create(
                        model=model,
                        max_tokens=50,
                        messages=[{"role": "user", "content": judge_prompt}],
                    )
                    if self._cost_callback and anthropic_result.usage:
                        self._cost_callback(
                            model,
                            anthropic_result.usage.input_tokens,
                            anthropic_result.usage.output_tokens,
                        )
                    first_block = (
                        anthropic_result.content[0]
                        if anthropic_result.content
                        else None
                    )
                    return (
                        first_block.text
                        if first_block is not None
                        and hasattr(first_block, "text")
                        else ""
                    )
                return None
            except ImportError as exc:
                logger.warning("LLM judge import failed: %s", exc)
                return None
            except Exception as exc:
                last_exc = exc
                if attempt < len(self._JUDGE_RETRY_BACKOFF):
                    time.sleep(self._JUDGE_RETRY_BACKOFF[attempt])

        logger.warning(
            "LLM judge failed after %d attempts: %s",
            self._JUDGE_RETRY_MAX,
            last_exc,
        )
        return None

    @staticmethod
    def _parse_judge_reply(reply: str) -> tuple[bool, float]:
        """Parse verdict and confidence from LLM judge JSON.

        Returns (agrees: bool, confidence: 0.0-1.0).
        Falls back to string matching with 0.5 confidence.
        """
        import json as _json

        try:
            data = _json.loads(reply)
            agrees = str(data.get("verdict", "")).upper() == "YES"
            raw_conf = float(data.get("confidence", 50))
            conf = max(0.0, min(1.0, raw_conf / 100.0))
            return agrees, conf
        except (ValueError, TypeError, AttributeError):
            return "YES" in reply.upper(), 0.5
