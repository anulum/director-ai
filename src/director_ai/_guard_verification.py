# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production Guard Extended Verification
"""Extended verification surface of the production guard.

:class:`ExtendedVerificationMixin` carries the verification modalities
of :class:`~director_ai.guard.ProductionGuard` beyond the core
prompt/response coherence check: temporal claim consistency and live
temporal refresh, token-level hallucinated-span detection, multimodal
(image/audio/video) evidence verification, neuro-symbolic SMT
compliance, and LTL trajectory safety monitoring. Stateful verifiers
are built lazily on first use and persist on the guard; the
advanced-tier modules are imported inside the methods so the Apache
core wheel does not require them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from director_ai.core.config import DirectorConfig
    from director_ai.core.multimodal_guard import (
        MultimodalCheckRequest,
        MultimodalCheckResult,
        MultimodalVerifierAdapter,
    )
    from director_ai.core.neuro_symbolic import (
        CompliancePolicy,
        NeuroSymbolicComplianceEngine,
    )
    from director_ai.core.scoring.span_detector import (
        HallucinationSpanDetector,
        SpanDetection,
    )
    from director_ai.core.scoring.temporal_refresh import (
        RefreshReport,
        TemporalRefresher,
    )
    from director_ai.core.temporal_consistency import TemporalConsistencyGraph
    from director_ai.core.temporal_logic import TrajectorySafetyMonitor

__all__ = ["ExtendedVerificationMixin"]


class ExtendedVerificationMixin:
    """Temporal, span, multimodal, compliance, and trajectory verification.

    All state is initialised by :class:`~director_ai.guard.ProductionGuard`'s
    ``__init__``; the configuration comes from the composing guard through the
    ``_config`` contract declared below.
    """

    _temporal_consistency: TemporalConsistencyGraph | None
    _temporal_refresher: TemporalRefresher | None
    _multimodal: MultimodalVerifierAdapter | None
    _span_detector: HallucinationSpanDetector | None

    if TYPE_CHECKING:
        # Provided by the composing ProductionGuard.
        _config: DirectorConfig

    @property
    def temporal_consistency(self) -> TemporalConsistencyGraph:
        """Cross-session structured-claim temporal consistency graph.

        Persists across calls on this guard so claims recorded in one session are
        checked against earlier sessions — the formal answer to "the system said
        X yesterday and ¬X today". Record
        :class:`~director_ai.core.temporal_consistency.TemporalClaim` objects and
        read contradictions back. For single-valued predicates (one diagnosis per
        patient) construct
        :class:`~director_ai.core.temporal_consistency.TemporalConsistencyGraph`
        directly with ``functional_predicates``.
        """
        if self._temporal_consistency is None:
            try:
                from director_ai.core.temporal_consistency import (
                    TemporalConsistencyGraph,
                )
            except ModuleNotFoundError as exc:  # pragma: no cover - advanced tier only
                raise RuntimeError(
                    "temporal consistency requires the advanced tier "
                    "(director_ai.core.temporal_consistency is not installed). "
                    "Install director-ai-pro to use it."
                ) from exc

            self._temporal_consistency = TemporalConsistencyGraph()
        return self._temporal_consistency

    @property
    def temporal_refresher(self) -> TemporalRefresher:
        """Live web-search refresher for stale temporal claims.

        Where :attr:`temporal_consistency` checks claims against this system's own
        history, the refresher checks them against the live web: it scores a
        response for staleness, then for each flagged claim (a named office-holder,
        a statistic, a record) queries a search provider and reports whether
        current sources still support it. Persists across calls. By default it runs
        the dependency-free lexical triage; inject an NLI
        :class:`~director_ai.core.scoring.contradiction.ContradictionScorer` via
        ``TemporalRefresher(nli=...)`` for adjudicated ``contradicted`` verdicts.
        """
        if self._temporal_refresher is None:
            try:
                from director_ai.core.scoring.temporal_refresh import (
                    TemporalRefresher,
                )
            except ModuleNotFoundError as exc:  # pragma: no cover - advanced tier only
                raise RuntimeError(
                    "live temporal refresh requires the advanced tier "
                    "(director_ai.core.scoring.temporal_refresh is not installed). "
                    "Install director-ai-pro to use it."
                ) from exc

            self._temporal_refresher = TemporalRefresher()
        return self._temporal_refresher

    def refresh_temporal(
        self,
        text: str,
        *,
        source_timestamp: float | None = None,
        max_age_days: float = 180,
        domain: str = "",
    ) -> RefreshReport:
        """Score *text* for staleness and live-refresh its stale claims.

        Convenience wrapper over :attr:`temporal_refresher` that returns a
        :class:`~director_ai.core.scoring.temporal_refresh.RefreshReport`.
        """
        return self.temporal_refresher.refresh_response(
            text,
            source_timestamp=source_timestamp,
            max_age_days=max_age_days,
            domain=domain,
        )

    @property
    def multimodal_adapter(self) -> MultimodalVerifierAdapter:
        """In-process multimodal verifier (created on first use).

        Builds the dependency-free hash-bag
        :class:`~director_ai.core.multimodal_guard.MultimodalVerifierAdapter` from
        the guard's ``multimodal_*`` config. Requires
        ``multimodal_enabled_modalities`` to be set (the guard is opt-in); raises
        otherwise. Pass a torch/CLIP-backed adapter to
        :meth:`check_multimodal` for semantic verification.
        """
        if self._multimodal is None:
            cfg = self._config
            if not cfg.multimodal_enabled_modalities:
                raise RuntimeError(
                    "multimodal guard is disabled; set "
                    "multimodal_enabled_modalities in the config to enable it"
                )
            if cfg.multimodal_backend == "clip":
                from director_ai.core.multimodal_guard import build_clip_adapter

                self._multimodal = build_clip_adapter(
                    enabled_modalities=cfg.multimodal_enabled_modalities,
                    benchmarked_modalities=cfg.multimodal_benchmarked_modalities,
                    model_name=cfg.multimodal_clip_model,
                    pretrained=cfg.multimodal_clip_pretrained,
                    device=cfg.multimodal_clip_device,
                    text_dim=cfg.multimodal_embedding_dim,
                    hallucination_threshold=cfg.multimodal_hallucination_threshold,
                    consistency_threshold=cfg.multimodal_consistency_threshold,
                    temporal_alpha=cfg.multimodal_temporal_alpha,
                    temporal_floor=cfg.multimodal_temporal_floor,
                    grounding_floor=cfg.multimodal_grounding_floor,
                    grounding_allow_threshold=cfg.multimodal_grounding_allow_threshold,
                )
            elif cfg.multimodal_backend == "hashbag":
                from director_ai.core.multimodal_guard import build_hashbag_adapter

                self._multimodal = build_hashbag_adapter(
                    enabled_modalities=cfg.multimodal_enabled_modalities,
                    benchmarked_modalities=cfg.multimodal_benchmarked_modalities,
                    dim=cfg.multimodal_embedding_dim,
                    hallucination_threshold=cfg.multimodal_hallucination_threshold,
                    consistency_threshold=cfg.multimodal_consistency_threshold,
                    temporal_alpha=cfg.multimodal_temporal_alpha,
                    temporal_floor=cfg.multimodal_temporal_floor,
                    grounding_floor=cfg.multimodal_grounding_floor,
                    grounding_allow_threshold=cfg.multimodal_grounding_allow_threshold,
                )
            else:
                raise RuntimeError(
                    f"unknown multimodal_backend {cfg.multimodal_backend!r}; "
                    "expected 'hashbag' or 'clip'"
                )
        return self._multimodal

    @property
    def span_detector(self) -> HallucinationSpanDetector:
        """Token-level hallucinated-span detector (created on first use).

        Loads the ModernBERT token classifier named by ``span_model`` and flags
        the unsupported spans inside a RAG response — the span-level signal the
        response/claim-level scorer cannot isolate. Opt-in: requires
        ``span_detection_enabled`` in the config; raises otherwise.
        """
        if self._span_detector is None:
            cfg = self._config
            if not cfg.span_detection_enabled:
                raise RuntimeError(
                    "span detection is disabled; set span_detection_enabled in "
                    "the config to enable it"
                )
            from director_ai.core.scoring.span_detector import (
                HallucinationSpanDetector,
            )

            self._span_detector = HallucinationSpanDetector.from_pretrained(
                cfg.span_model,
                revision=cfg.span_model_revision or None,
                device=cfg.span_device,
                token_threshold=cfg.span_token_threshold,
                min_tokens=cfg.span_min_tokens,
                max_length=cfg.span_max_length,
            )
        return self._span_detector

    def detect_spans(self, context: str, response: str) -> SpanDetection:
        """Flag the hallucinated spans of *response* against *context*.

        Returns a
        :class:`~director_ai.core.scoring.span_detector.SpanDetection` with the
        character spans the token detector judged unsupported. Requires
        ``span_detection_enabled``.
        """
        return self.span_detector.detect(context, response)

    def check_multimodal(
        self,
        request: MultimodalCheckRequest,
        *,
        adapter: MultimodalVerifierAdapter | None = None,
    ) -> MultimodalCheckResult:
        """Verify a text claim against paired image / audio / video evidence.

        The in-process counterpart of the ``/v1/multimodal/check`` endpoint:
        scores a
        :class:`~director_ai.core.multimodal_guard.MultimodalCheckRequest` with
        the configured (or supplied) adapter and returns a tenant-safe
        :class:`~director_ai.core.multimodal_guard.MultimodalCheckResult` carrying
        the shared allow/warn/halt decision. Modalities enabled but not in
        ``multimodal_benchmarked_modalities`` resolve to ``warn`` rather than a
        scored decision, so an unbenchmarked modality never silently passes.
        """
        from director_ai.core.guard_control import RiskEnvelope

        cfg = self._config
        used = adapter if adapter is not None else self.multimodal_adapter
        risk_envelope = RiskEnvelope(
            action_category="multimodal",
            reversibility="reversible",
            domain="general",
            calibrated_threshold=getattr(cfg, "multimodal_calibrated_threshold", 0.5),
            no_go_threshold=getattr(cfg, "multimodal_no_go_threshold", 0.9),
        )
        policy_id = getattr(cfg, "multimodal_policy_id", "multimodal-default")
        return used.check(request, risk_envelope=risk_envelope, policy_id=policy_id)

    def compliance_engine(
        self, policy: CompliancePolicy
    ) -> NeuroSymbolicComplianceEngine:
        """Build a neuro-symbolic SMT compliance engine for ``policy``.

        ``policy`` is a
        :class:`~director_ai.core.neuro_symbolic.CompliancePolicy`. The engine
        checks structured output facts against the policy with Z3, returning a
        per-constraint verdict with counterexamples, and can cross-check that two
        formalisations are equivalent. Requires the ``[formal]`` extra (z3); the
        engine raises a clear error if z3 is unavailable.
        """
        from director_ai.core.neuro_symbolic import NeuroSymbolicComplianceEngine

        return NeuroSymbolicComplianceEngine(policy)

    def trajectory_monitor(
        self, specs: dict[str, object] | None = None
    ) -> TrajectorySafetyMonitor:
        """Return a fresh LTL safety monitor for one agent trajectory.

        Runs the built-in agent-safety specifications (tool calls eventually
        verified, handoffs coherence-checked, no output after an injection, fact
        claims eventually grounded) — the formal reading of EU AI Act Article 15
        "continuous monitoring". Feed one
        :class:`~director_ai.core.temporal_logic.StepObservation` per trajectory
        step; pass ``specs`` to override or extend the default set. Each call
        returns an independent monitor, so concurrent trajectories do not share
        state.
        """
        from director_ai.core.temporal_logic import TrajectorySafetyMonitor

        return TrajectorySafetyMonitor(specs)
