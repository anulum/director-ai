# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Review Pipeline (approve/reject orchestration mixin)
"""Review orchestration for the coherence scorer.

:class:`ReviewPipelineMixin` drives a full review round on top of the
divergence signals from :class:`~director_ai.core.scoring._divergence.
DivergenceMixin`: cache lookup and scoping, cross-turn blending, adaptive
and meta-classifier thresholds, verdict finalisation, the verified-claim
and Tier-6 reasoning escalations, injection detection hand-off, the
coalesced batch path, and the async wrapper. All state is initialised by
the composing scorer's ``__init__``; the ``TYPE_CHECKING`` stubs document
the scorer-provided services the mixin calls.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

from ..otel import trace_cache, trace_calibration, trace_review
from ..types import CoherenceScore, ScoringEvidence
from ._divergence import DIVERGENCE_NEUTRAL, DivergenceMixin

if TYPE_CHECKING:
    from ..cache import ScoreCache
    from ..calibration.conformal import ConformalPredictor
    from ..redactor import PIIRedactor
    from ._llm_judge import LLMJudge
    from .reasoning_scorer import ReasoningScorer
    from .self_consistency import SelfConsistencyScorer

__all__ = ["ReviewPipelineMixin"]


class ReviewPipelineMixin(DivergenceMixin):
    """Review-orchestration surface of :class:`CoherenceScorer`.

    Builds on the divergence signals to score, gate, and finalise a
    review: caching, thresholds (static, adaptive, meta-classifier),
    verified-claim and reasoning escalations, session bookkeeping, and
    the batch/async entry points. All state is initialised by the
    composing scorer's ``__init__``; the annotations below declare that
    shared contract for static analysis without creating attributes.
    """

    # Shared state initialised by the composing scorer.
    threshold: float
    soft_limit: float
    history: list[str]
    window: int
    cache: ScoreCache | None
    _history_lock: threading.Lock
    _dry_run: bool
    _adaptive_threshold_enabled: bool
    _task_type_thresholds: dict[str, float]
    _verified_scorer_enabled: bool
    _verified_scorer_atomic: bool
    _verified_scorer_evidence_top_k: int
    _verified_scorer_low_confidence_margin: float
    _verified_scorer_min_coverage: float
    _verified_scorer_task_types: set[str]
    _conformal_predictor: ConformalPredictor | None
    _self_consistency_scorer: SelfConsistencyScorer | None
    _self_consistency_weight: float
    _judge: LLMJudge
    _reasoning: ReasoningScorer
    _redactor: PIIRedactor

    if TYPE_CHECKING:
        # Services provided by the composing scorer.

        def _enforce_model_backed_nli_requirement(self) -> None: ...

        def _get_meta_classifier(self) -> Any | None: ...

        def _get_injection_runtime_state(self) -> tuple[Any | None, bool]: ...

    def enable_conformal(
        self,
        predictor: ConformalPredictor | None = None,
        *,
        coverage: float = 0.95,
        min_samples: int = 30,
    ) -> ConformalPredictor:
        """Opt in to conformal hallucination-risk intervals on ``review()``.

        Attaches a :class:`~director_ai.core.calibration.conformal.
        ConformalPredictor` (a supplied one, or a fresh predictor at the
        given *coverage* and *min_samples*) and returns it so the caller
        can calibrate it — ``calibrate()``, ``calibrate_from_feedback()``,
        or per-observation ``add_observation()``. Until enabled, review
        results carry no conformal fields and behaviour is unchanged.
        """
        if predictor is None:
            from ..calibration.conformal import ConformalPredictor

            predictor = ConformalPredictor(coverage=coverage, min_samples=min_samples)
        self._conformal_predictor = predictor
        return predictor

    def _apply_conformal_interval(
        self,
        result: tuple[bool, CoherenceScore],
    ) -> tuple[bool, CoherenceScore]:
        """Attach the calibrated risk interval to a finalised review result.

        A no-op until :meth:`enable_conformal` is called, so the default
        review path is unchanged. The interval bounds P(hallucination) at
        the predictor's coverage level; approval is not altered — routing
        on the bounds stays with the caller (see ``ConformalRoutingPolicy``).
        """
        predictor = self._conformal_predictor
        if predictor is None:
            return result
        interval = predictor.predict(result[1].score)
        score = result[1]
        score.conformal_risk_lower = interval.lower
        score.conformal_risk_upper = interval.upper
        score.conformal_coverage = interval.coverage
        score.conformal_calibration_size = interval.calibration_size
        score.conformal_reliable = interval.is_reliable
        return result

    def enable_self_consistency(
        self,
        scorer: SelfConsistencyScorer | None = None,
        *,
        weight: float = 0.25,
    ) -> SelfConsistencyScorer:
        """Opt in to the semantic-entropy signal on ``review_with_samples()``.

        Attaches a :class:`~director_ai.core.scoring.self_consistency.
        SelfConsistencyScorer` (a supplied one, or a fresh scorer reusing
        this scorer's NLI backend when model-backed) and the fusion
        *weight* in [0, 1). Until enabled, ``review_with_samples()``
        raises and ``review()`` behaviour is unchanged.
        """
        if not 0.0 <= weight < 1.0:
            raise ValueError("weight must be in [0, 1)")
        if scorer is None:
            from .self_consistency import SelfConsistencyScorer

            nli = self._nli if getattr(self, "use_nli", False) else None
            scorer = SelfConsistencyScorer(nli_scorer=nli)
        self._self_consistency_scorer = scorer
        self._self_consistency_weight = weight
        return scorer

    def review_with_samples(
        self,
        prompt: str,
        action: str,
        samples: list[str],
        session: Any | None = None,
        tenant_id: str = "",
    ) -> tuple[bool, CoherenceScore]:
        """``review()`` fused with the sample-consistency signal.

        Runs the standard review of ``action``, scores its semantic
        consistency against caller-supplied alternative ``samples``
        (same prompt, independent generations), then blends:
        ``fused = (1 − w)·review_score + w·consistency_score`` and
        re-gates approval on the same threshold the plain review used
        (a fused score below it revokes approval; fusion never
        approves what ``review()`` rejected). The consistency fields
        are attached to the returned :class:`CoherenceScore`.

        Requires a prior :meth:`enable_self_consistency` call —
        consistency with zero samples is not evidence, so there is no
        silent fallback.

        Raw-support routes (WCS-2a) attach the consistency fields but do
        NOT blend them into the score: their coherence is a raw support
        gated at a matched-FPR operating point, and averaging it with a
        composite-scale consistency score would silently move that
        calibrated operating point.
        """
        if self._self_consistency_scorer is None:
            raise RuntimeError(
                "call enable_self_consistency() before review_with_samples()",
            )
        approved, score = self.review(
            prompt,
            action,
            session=session,
            tenant_id=tenant_id,
        )
        consistency = self._self_consistency_scorer.score(action, samples)

        score.self_consistency_score = round(consistency.consistency_score, 4)
        score.semantic_entropy = round(consistency.semantic_entropy, 4)
        score.self_consistency_backend = consistency.entailment_backend
        if self._raw_support_operating_point(prompt, action) is not None:
            return approved, score

        weight = self._self_consistency_weight
        fused = (1.0 - weight) * score.score + weight * consistency.consistency_score
        score.score = round(fused, 4)
        if approved and fused < self.threshold:
            approved = False
            score.approved = False
        return approved, score

    def _effective_review_threshold(
        self,
        prompt: str,
        action: str,
        task_type: str,
        raw_op: float | None = None,
    ) -> tuple[float, float | None]:
        """Resolve the review gate for one input.

        Returns ``(threshold, soft_limit_override)``. A raw-support route
        (WCS-2a) gates on its matched-FPR support operating point and
        carries the configured soft-limit margin onto the support scale;
        otherwise the adaptive per-task-type threshold and the
        meta-classifier override apply on the composite-coherence scale,
        with ``soft_limit_override=None`` (the configured soft limit is
        already on that scale).
        """
        if raw_op is None:
            raw_op = self._raw_support_operating_point(prompt, action)
        if raw_op is not None:
            margin = max(0.0, self.soft_limit - self.threshold)
            return raw_op, min(1.0, raw_op + margin)

        threshold = self.threshold
        if self._adaptive_threshold_enabled and self._task_type_thresholds:
            threshold = self._task_type_thresholds.get(task_type, self.threshold)

        # Meta-classifier: dataset-type mode predicts which sub-dataset
        # the input resembles, then applies the optimal NLI threshold
        # for that dataset. Falls back to per-task-type if uncertain.
        meta_clf = self._get_meta_classifier()
        if meta_clf is not None:
            nli_threshold, _meta_conf = meta_clf.predict_threshold(prompt, action)
            if nli_threshold is not None:
                # NLI-scale to coherence-scale
                threshold = self.W_FACT + self.W_LOGIC * nli_threshold
        return threshold, None

    def _finalise_review(
        self,
        coherence: float,
        h_logic: float,
        h_fact: float,
        action: str,
        evidence: ScoringEvidence | None = None,
        threshold_override: float | None = None,
        detected_task_type: str | None = None,
        escalated_to_judge: bool | None = None,
        soft_limit_override: float | None = None,
    ) -> tuple[bool, CoherenceScore]:
        """Build CoherenceScore, gate on threshold, update history.

        Returns (approved, CoherenceScore).
        """
        t = threshold_override if threshold_override is not None else self.threshold
        approved = coherence >= t

        # Dry-run mode: log actual score but always approve
        if self._dry_run and not approved:
            self.logger.info(
                "DRY-RUN: would reject (score=%.3f < threshold=%.3f) but approving",
                coherence,
                t,
            )
            approved = True
        warning = False

        if not approved:
            self.logger.critical(
                "COHERENCE FAILURE. Score: %.4f < Threshold: %s",
                coherence,
                t,
            )
        else:
            soft = (
                soft_limit_override
                if soft_limit_override is not None
                else self.soft_limit
            )
            if coherence < soft:
                warning = True
            with self._history_lock:
                self.history.append(action)
                if len(self.history) > self.window:
                    self.history.pop(0)

        strict_rejected = self.strict_mode and not (
            self._nli and self._nli.model_available
        )

        from .meta_confidence import compute_meta_confidence

        with trace_calibration(stage="meta_confidence") as calibration_span:
            vc, _mc, sa = compute_meta_confidence(
                score=coherence,
                threshold=t,
                h_logical=h_logic,
                h_factual=h_fact,
            )
            calibration_span.set_attribute("calibration.threshold", t)
            calibration_span.set_attribute("calibration.verdict_confidence", vc)
            calibration_span.set_attribute("calibration.signal_agreement", sa)

        # Retrieval confidence from evidence chunks
        retrieval_conf = None
        if evidence is not None and evidence.chunks:
            best = min((c.distance for c in evidence.chunks), default=1.0)
            retrieval_conf = max(0.0, 1.0 - best)

        score = CoherenceScore(
            score=coherence,
            approved=approved,
            h_logical=h_logic,
            h_factual=h_fact,
            evidence=evidence,
            warning=warning,
            strict_mode_rejected=strict_rejected,
            verdict_confidence=vc,
            signal_agreement=sa,
            detected_task_type=detected_task_type,
            escalated_to_judge=escalated_to_judge,
            retrieval_confidence=retrieval_conf,
        )
        return approved, score

    def _verified_source_from_evidence(self, evidence: ScoringEvidence | None) -> str:
        """Build source text for claim-level verification from scoring evidence."""
        if evidence is None:
            return ""
        chunk_texts = [
            chunk.text.strip() for chunk in evidence.chunks if chunk.text.strip()
        ]
        if chunk_texts:
            return " ".join(chunk_texts)
        return evidence.nli_premise.strip()

    def _should_run_verified_scorer(
        self,
        *,
        coherence: float,
        threshold: float,
        task_type: str,
        evidence: ScoringEvidence | None,
    ) -> bool:
        """Return whether atomic verification should run on the review path."""
        if not self._verified_scorer_enabled:
            return False
        if not self._verified_source_from_evidence(evidence):
            return False
        low_confidence = (
            abs(coherence - threshold) <= self._verified_scorer_low_confidence_margin
        )
        task_routed = task_type in self._verified_scorer_task_types
        return low_confidence or task_routed

    def _apply_verified_scorer(
        self,
        score: CoherenceScore,
        *,
        task_type: str,
        threshold: float,
    ) -> tuple[bool, CoherenceScore]:
        """Attach atomic verification and fail closed on verified claim failures."""
        if not self._should_run_verified_scorer(
            coherence=score.score,
            threshold=threshold,
            task_type=task_type,
            evidence=score.evidence,
        ):
            return score.approved, score

        source = self._verified_source_from_evidence(score.evidence)
        from .verified_scorer import VerifiedScorer

        verifier = VerifiedScorer(nli_scorer=self._nli)
        result = verifier.verify(
            score.evidence.nli_hypothesis if score.evidence else "",
            source,
            atomic=self._verified_scorer_atomic,
            evidence_top_k=self._verified_scorer_evidence_top_k,
        )
        payload = result.to_dict()
        score.verified_result = payload
        score.verified_approved = result.approved
        score.verified_coverage = result.coverage
        score.verified_claim_count = len(result.claims)
        score.verified_contradicted_count = result.contradicted_count
        score.verified_fabricated_count = result.fabricated_count
        insufficient_coverage = (
            task_type in self._verified_scorer_task_types
            and len(result.claims) > 0
            and result.coverage < self._verified_scorer_min_coverage
        )
        if not result.approved or insufficient_coverage:
            score.verified_approved = False
            score.approved = False
            return False, score
        return score.approved, score

    def _apply_reasoning_tier(
        self,
        result: tuple[bool, CoherenceScore],
        prompt: str,
        action: str,
        evidence: ScoringEvidence | None,
        *,
        threshold: float,
    ) -> tuple[bool, CoherenceScore]:
        """Consult the Tier-6 reasoning escalation on a borderline verdict.

        Fires only when the composite score sits within the reasoning tier's
        margin of *threshold*. A parsed verdict blends into the score and tags
        the result with rationale + harm category; an unavailable backend or an
        unparsable reply leaves the lower-tier verdict untouched. Approval then
        requires both the blended score to clear the threshold *and* the
        reasoning verdict to approve, so a confident safety rejection halts a
        borderline output.
        """
        _approved, score = result
        if not self._reasoning.should_escalate(score.score, centre=threshold):
            return result
        source = self._verified_source_from_evidence(evidence) if evidence else ""
        if not isinstance(source, str):
            source = ""
        verdict = self._reasoning.reason(
            prompt,
            action,
            score.score,
            task_type=score.detected_task_type or "default",
            evidence_text=source,
            redactor=self._redactor,
        )
        if verdict is None:
            return result
        score.reasoning_escalated = True
        score.reasoning_confidence = verdict.confidence
        score.reasoning_rationale = verdict.rationale
        score.reasoning_harm_category = (
            verdict.harm_category.value if verdict.harm_category is not None else None
        )
        if verdict.adjusted_score is not None:
            score.score = verdict.adjusted_score
        new_approved = score.score >= threshold and verdict.approved
        score.approved = new_approved
        return new_approved, score

    # ── Composite scoring ─────────────────────────────────────────────

    def _score_cache_scope(
        self, session: Any | None = None, tenant_id: str = ""
    ) -> str:
        """Build cache scope from conversation and mutable grounding state."""
        scope_parts = []
        if session is not None and len(session) > 0:
            scope_parts.append(f"session:{session.context_text}")
        store = self.ground_truth_store
        if store is not None and hasattr(store, "cache_scope"):
            scope_parts.append(f"store:{store.cache_scope(tenant_id=tenant_id)}")
        return "\x1f".join(scope_parts)

    def compute_divergence(self, prompt: str, action: str) -> float:
        """Compute composite divergence (lower is better).

        Weighted sum: ``W_LOGIC * H_logical + W_FACT * H_factual``.
        """
        h_logic = self.calculate_logical_divergence(prompt, action)
        h_fact = self.calculate_factual_divergence(prompt, action)
        total = (self.W_LOGIC * h_logic) + (self.W_FACT * h_fact)
        self.logger.debug(
            "Divergence: Logic=%.2f, Fact=%.2f -> Total=%.2f",
            h_logic,
            h_fact,
            total,
        )
        return total

    def review(
        self,
        prompt: str,
        action: str,
        session: Any | None = None,
        tenant_id: str = "",
    ) -> tuple[bool, CoherenceScore]:
        """Score an action and decide whether to approve it.

        Parameters
        ----------
        prompt : str
            Source prompt, user request, or retrieved question that frames the
            response under review.
        action : str
            Candidate model output to score against the prompt and any
            configured grounding store.
        session : ConversationSession | None – when provided, cross-turn
            divergence is blended into the logical score and the turn is
            recorded after scoring.
        tenant_id : str
            Tenant scope for cache keys and tenant-aware grounding stores.

        """
        with trace_review() as span:
            self._enforce_model_backed_nli_requirement()
            # Rust fast-path: delegate full review to backfire_kernel
            if self._rust_scorer is not None:
                approved_r, score_obj = self._rust_scorer.review(prompt, action)
                h_l = getattr(score_obj, "h_logical", 0.0)
                h_f = getattr(score_obj, "h_factual", 0.0)
                fallback = 1.0 - (self.W_LOGIC * h_l + self.W_FACT * h_f)
                coh = getattr(score_obj, "score", fallback)
                result = self._apply_conformal_interval(
                    self._finalise_review(coh, h_l, h_f, action)
                )
                span.set_attribute("coherence.score", result[1].score)
                span.set_attribute("coherence.approved", result[0])
                span.set_attribute("coherence.backend", "rust")
                return result

            cache_scope = self._score_cache_scope(session=session, tenant_id=tenant_id)

            # Raw-support operating point (WCS-2a): resolved once so the
            # cache-hit and fresh-scoring paths gate identically — a
            # decision must not depend on cache state.
            raw_op = self._raw_support_operating_point(prompt, action)

            if self.cache:
                with trace_cache(scope_present=bool(cache_scope)) as cache_span:
                    cached = self.cache.get(
                        prompt,
                        action,
                        tenant_id=tenant_id,
                        scope=cache_scope,
                    )
                    cache_span.set_attribute("cache.hit", cached is not None)
                if cached is not None:
                    cached_task = self._detect_task_type(prompt, action)
                    cached_t, cached_soft = self._effective_review_threshold(
                        prompt,
                        action,
                        cached_task,
                        raw_op=raw_op,
                    )
                    result = self._apply_conformal_interval(
                        self._finalise_review(
                            cached.score,
                            cached.h_logical,
                            cached.h_factual,
                            action,
                            threshold_override=cached_t,
                            detected_task_type=cached_task,
                            soft_limit_override=cached_soft,
                        )
                    )
                    span.set_attribute("coherence.score", cached.score)
                    span.set_attribute("coherence.approved", result[0])
                    span.set_attribute("coherence.cached", True)
                    return result
            h_logic, h_fact, coherence, evidence = self._heuristic_coherence(
                prompt,
                action,
                tenant_id=tenant_id,
            )

            cross_turn = None
            # Raw-support routes also skip the cross-turn blend: their
            # coherence is a calibrated raw support, and reweighting it
            # with a cross-turn component would move the matched-FPR
            # operating point.
            _skip_cross_turn = raw_op is not None or (
                self._auto_dialogue_profile
                and not self._use_prompt_as_premise
                and self._nli is not None
                and self._nli.model_available
                and self._detect_task_type(prompt, action) == "dialogue"
            )
            if session is not None and len(session) > 0 and not _skip_cross_turn:
                ctx = session.context_text
                if ctx.strip() and self._nli:
                    cross_turn = self._nli.score(ctx, action)
                    h_logic = 0.7 * h_logic + 0.3 * cross_turn
                    total_divergence = self.W_LOGIC * h_logic + self.W_FACT * h_fact
                    coherence = 1.0 - total_divergence
                    # Re-apply no-KB calibration after cross-turn blend
                    nli_ok = self._nli is not None and self._nli.model_available
                    if (
                        nli_ok
                        and abs(h_fact - DIVERGENCE_NEUTRAL) < 1e-9
                        and evidence is None
                    ):
                        cal_lo = 1.0 - self.W_LOGIC - self.W_FACT * DIVERGENCE_NEUTRAL
                        cal_hi = 1.0 - self.W_FACT * DIVERGENCE_NEUTRAL
                        cal_span = cal_hi - cal_lo
                        if cal_span > 1e-9:
                            coherence = max(
                                0.0, min(1.0, (coherence - cal_lo) / cal_span)
                            )

            if self.cache:
                self.cache.put(
                    prompt,
                    action,
                    coherence,
                    h_logic,
                    h_fact,
                    tenant_id=tenant_id,
                    scope=cache_scope,
                )

            # Always detect task type for explainability
            task_type = self._detect_task_type(prompt, action)

            # Gate resolution: raw-support operating point, adaptive
            # per-task-type threshold, or meta-classifier override.
            effective_threshold, soft_override = self._effective_review_threshold(
                prompt,
                action,
                task_type,
                raw_op=raw_op,
            )

            result = self._finalise_review(
                coherence,
                h_logic,
                h_fact,
                action,
                evidence,
                threshold_override=effective_threshold,
                detected_task_type=task_type,
                soft_limit_override=soft_override,
            )
            result = self._apply_verified_scorer(
                result[1],
                task_type=task_type,
                threshold=effective_threshold,
            )
            if self._reasoning.enabled:
                result = self._apply_reasoning_tier(
                    result,
                    prompt,
                    action,
                    evidence,
                    threshold=effective_threshold,
                )
            if cross_turn is not None:
                result[1].cross_turn_divergence = cross_turn
            contradiction_trend = 0.0
            if session is not None:
                if self._nli and self._nli.model_available:
                    try:
                        report = session.update_contradictions(
                            action, lambda p, h: self._nli.score(p, h)
                        )
                        result[1].contradiction_index = report.contradiction_index
                        contradiction_trend = report.trend
                    except Exception:
                        self.logger.warning(
                            "Contradiction tracking failed", exc_info=True
                        )
                session.add_turn(prompt, action, result[1].score)
            # Injection detection (when enabled)
            inj_detector, inj_fail_closed = self._get_injection_runtime_state()
            if inj_detector is not None:
                try:
                    inj = inj_detector.detect(intent=prompt, response=action)
                    result[1].injection_risk = inj.injection_risk
                except Exception:
                    if inj_fail_closed:
                        raise
                    self.logger.warning("Injection detection failed", exc_info=True)
            # Long-context intent-drift interlock (opt-in on the session)
            interlock = getattr(session, "intent_drift", None)
            if interlock is not None:
                drift = interlock.update(
                    intent_divergence=cross_turn if cross_turn is not None else 0.0,
                    injection_risk=result[1].injection_risk or 0.0,
                    contradiction_trend=contradiction_trend,
                )
                result[1].intent_drift_risk = drift.drift_risk
                result[1].intent_drift_triggered = drift.triggered

            result = self._apply_conformal_interval(result)
            span.set_attribute("coherence.score", result[1].score)
            span.set_attribute("coherence.approved", result[0])
            span.set_attribute("coherence.cached", False)
            span.set_attribute("coherence.h_logical", h_logic)
            span.set_attribute("coherence.h_factual", h_fact)
            span.set_attribute("coherence.warning", result[1].warning)
            if result[1].injection_risk is not None:
                span.set_attribute("coherence.injection_risk", result[1].injection_risk)
            return result

    # ── Batch API (coalesced NLI) ────────────────────────────────────

    def review_batch(
        self,
        items: list[tuple[str, str]],
        tenant_id: str = "",
    ) -> list[tuple[bool, CoherenceScore]]:
        """Batch-review a list of (prompt, response) pairs.

        When NLI is available, batches logical and factual divergence
        through ``NLIScorer.score_batch()`` (2 GPU forward passes total
        instead of 2*N).  Falls back to sequential ``review()`` for
        items that need special handling (dialogue, summarization, rust
        backend, or when NLI is unavailable).
        """
        self._enforce_model_backed_nli_requirement()
        if not items:
            return []
        nli_ok = (
            self._nli is not None
            and self._nli.model_available
            and self._rust_scorer is None
            and not self._use_prompt_as_premise
        )
        if not nli_ok or len(items) < 2:
            return [self.review(p, a, tenant_id=tenant_id) for p, a in items]
        if self._review_batch_requires_sequential(items):
            return [self.review(p, a, tenant_id=tenant_id) for p, a in items]

        # Partition: batchable (standard path) vs fallback (dialogue etc.)
        batch_idx: list[int] = []
        fallback_idx: list[int] = []
        for i, (prompt, _action) in enumerate(items):
            if self._auto_dialogue_profile and self._detect_task_type(
                prompt, _action
            ) in ("dialogue", "summarization"):
                fallback_idx.append(i)
            else:
                batch_idx.append(i)

        results: list[tuple[bool, CoherenceScore] | None] = [None] * len(items)

        # Sequential fallback for dialogue/special items
        for i in fallback_idx:
            results[i] = self.review(items[i][0], items[i][1], tenant_id=tenant_id)

        if not batch_idx:
            return [r for r in results if r is not None]

        # Coalesced NLI. Retrieve the grounding context once per item first,
        # because it is the premise for BOTH the factual pass AND the logical
        # pass. A bare interrogative prompt is a degenerate NLI premise for the
        # logical signal — a true declarative answer does not entail the
        # question, inflating h_logical and false-halting true inputs (KIMI2-K).
        # Scoring the logical signal against the context keeps review_batch() in
        # parity with review() and fixes the false-halt on both paths.
        if self._nli is None:
            raise RuntimeError("NLI batch scorer not initialised")
        contexts: list[str | None] = []
        for i in batch_idx:
            prompt = items[i][0]
            if self.ground_truth_store and self._has_grounding_query(prompt):
                contexts.append(
                    self.ground_truth_store.retrieve_context(
                        prompt,
                        top_k=self._fact_retrieval_top_k,
                        tenant_id=tenant_id,
                    )
                    or None
                )
            else:
                contexts.append(None)

        # Logical: premise = grounding context when present, else the prompt.
        logic_pairs = [
            (contexts[pos] or items[i][0], items[i][1])
            for pos, i in enumerate(batch_idx)
        ]
        h_logics = self._nli.score_batch(logic_pairs)
        if len(h_logics) != len(logic_pairs):
            raise RuntimeError(
                "logical NLI batch returned "
                f"{len(h_logics)} scores for {len(logic_pairs)} pairs",
            )

        # Factual: batch the context-grounded items.
        h_facts: list[float] = []
        evidences: list[ScoringEvidence | None] = []
        fact_pairs: list[tuple[str, str]] = []
        fact_pair_map: list[int] = []  # maps fact_pairs index → batch position
        for pos, i in enumerate(batch_idx):
            ctx = contexts[pos]
            if ctx and self._nli:
                fact_pairs.append((ctx, items[i][1]))
                fact_pair_map.append(pos)
            h_facts.append(DIVERGENCE_NEUTRAL)
            evidences.append(None)

        if fact_pairs:
            fact_scores = self._nli.score_batch(fact_pairs)
            if len(fact_scores) != len(fact_pairs):
                raise RuntimeError(
                    "factual NLI batch returned "
                    f"{len(fact_scores)} scores for {len(fact_pairs)} pairs",
                )
            for j, fs in enumerate(fact_scores):
                h_facts[fact_pair_map[j]] = fs

        # Assemble coherence scores
        nli_available = True
        for pos, i in enumerate(batch_idx):
            h_logic = h_logics[pos]
            h_fact = h_facts[pos]
            evidence = evidences[pos]
            coherence = 1.0 - (self.W_LOGIC * h_logic + self.W_FACT * h_fact)

            # No-KB calibration
            fact_is_neutral = abs(h_fact - DIVERGENCE_NEUTRAL) < 1e-9
            if nli_available and fact_is_neutral and evidence is None:
                lo = 1.0 - self.W_LOGIC - self.W_FACT * DIVERGENCE_NEUTRAL
                hi = 1.0 - self.W_FACT * DIVERGENCE_NEUTRAL
                span = hi - lo
                if span > 1e-9:
                    coherence = max(0.0, min(1.0, (coherence - lo) / span))

            # Match review() finalisation: task-type, adaptive threshold,
            # meta-classifier — ensures batch/single parity. Dialogue and
            # summarisation items fall back to sequential review(), so
            # the raw-support operating point resolves to None here.
            prompt = items[i][0]
            task_type = self._detect_task_type(prompt, items[i][1])
            effective_threshold, soft_override = self._effective_review_threshold(
                prompt,
                items[i][1],
                task_type,
            )

            results[i] = self._apply_conformal_interval(
                self._finalise_review(
                    coherence,
                    h_logic,
                    h_fact,
                    items[i][1],
                    evidence,
                    threshold_override=effective_threshold,
                    detected_task_type=task_type,
                    soft_limit_override=soft_override,
                )
            )

        return [r for r in results if r is not None]

    def _review_batch_requires_sequential(
        self,
        items: list[tuple[str, str]],
    ) -> bool:
        """Return True when the coalesced path cannot match review() semantics."""
        if self._adaptive_router is not None:
            return True
        if self._retrieval_abstention_threshold > 0:
            return True
        if self._judge.enabled:
            return True
        if self._confidence_weighted_agg:
            return True
        if (
            self._fact_inner_agg != "max"
            or self._fact_outer_agg != "max"
            or self._logic_inner_agg != "max"
            or self._logic_outer_agg != "max"
        ):
            return True
        if self._rag_claim_decomposition and any(
            len(action) > 100 for _prompt, action in items
        ):
            return True
        if self.ground_truth_store is not None:
            from ..retrieval.vector_store import VectorGroundTruthStore

            if isinstance(self.ground_truth_store, VectorGroundTruthStore):
                return True
        return False

    # ── Async API ──────────────────────────────────────────────────────

    async def areview(
        self,
        prompt: str,
        action: str,
        session: Any | None = None,
        tenant_id: str = "",
    ) -> tuple[bool, CoherenceScore]:
        """Async version of review() – offloads NLI inference to a thread pool."""
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.review,
                prompt,
                action,
                session=session,
                tenant_id=tenant_id,
            ),
        )
