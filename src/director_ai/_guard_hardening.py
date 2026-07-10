# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production Guard Runtime Hardening
"""Runtime and supply-chain hardening surface of the production guard.

:class:`RuntimeHardeningMixin` carries the security-hardening
capabilities of :class:`~director_ai.guard.ProductionGuard`: the
agent/MCP preflight seam gates, zero-trust output encoding, graduated
execution rings, cryptographic output integrity, the ML bill of
materials, runtime self-protection, the guard-bypass fuzzer, threat
intelligence matching, and the embodied-robot command guard. Stateful
guards are built lazily on first use and persist on the guard; the
advanced-tier modules are imported inside the methods so the Apache
core wheel does not require them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from director_ai.core.agent_preflight import AgentPreflightGuard

if TYPE_CHECKING:
    from director_ai.core import CoherenceScorer
    from director_ai.core.cyber_physical import (
        PhysicalConstraint,
        RobotCommandGuard,
    )
    from director_ai.core.execution_rings import ExecutionRingGate
    from director_ai.core.fuzzing import ContinuousFuzzer
    from director_ai.core.ml_bom import MachineLearningBOM
    from director_ai.core.output_integrity import OutputIntegrityGuard
    from director_ai.core.output_trust import ZeroTrustOutputGuard
    from director_ai.core.rasp import RuntimeSelfProtection
    from director_ai.core.threat_intel import ThreatIntelligenceMatcher

__all__ = ["RuntimeHardeningMixin"]


class RuntimeHardeningMixin:
    """Preflight gates, output hardening, supply-chain and runtime defence.

    All state is initialised by :class:`~director_ai.guard.ProductionGuard`'s
    ``__init__``; the coherence scorer comes from the composing guard through
    the ``_scorer`` contract declared below.
    """

    _preflight: AgentPreflightGuard | None
    _output_trust: ZeroTrustOutputGuard | None
    _execution_rings: ExecutionRingGate | None
    _output_integrity: OutputIntegrityGuard | None
    _ml_bom: MachineLearningBOM | None
    _rasp: RuntimeSelfProtection | None
    _threat_intel: ThreatIntelligenceMatcher | None

    if TYPE_CHECKING:
        # Provided by the composing ProductionGuard.
        _scorer: CoherenceScorer

    @property
    def preflight(self) -> AgentPreflightGuard:
        """Agent/MCP preflight guard wired to this guard's scorer.

        Provides the five seam gates (before/after tool call, before final
        answer, before handoff, before irreversible action); result plausibility
        is scored with this guard's coherence scorer.
        """
        if self._preflight is None:

            def _score(premise: str, hypothesis: str) -> float:
                return self._scorer.review(premise, hypothesis)[1].score

            self._preflight = AgentPreflightGuard(score_fn=_score)
        return self._preflight

    @property
    def output_trust(self) -> ZeroTrustOutputGuard:
        """Zero-trust output handling for untrusted model output (OWASP LLM05).

        Encodes a model output for the specific
        :class:`~director_ai.core.output_trust.OutputSink` it will enter (HTML,
        shell argument, SQL identifier, filesystem path, JSON, URL query, email
        header, log line) so one string cannot be an XSS payload in one context
        and a command-injection payload in another, and flags constructs that
        must never be executed or deserialised. Stateless; persists on the guard
        only to avoid re-instantiation.
        """
        if self._output_trust is None:
            from director_ai.core.output_trust import ZeroTrustOutputGuard

            self._output_trust = ZeroTrustOutputGuard()
        return self._output_trust

    def execution_rings(
        self, *, cooling_period_seconds: float = 86_400.0
    ) -> ExecutionRingGate:
        """Graduated human authorisation for agent actions (execution rings).

        Classifies an action into an ordered
        :class:`~director_ai.core.execution_rings.ExecutionRing` (read → write →
        delete → execute → exfiltrate) and allows it only when the human
        authorisation factors that ring demands (operator approval, cooling
        period, second operator, CISO notification) have been collected — so a
        prompt-injected agent cannot delete or exfiltrate without out-of-band
        confirmation. ``cooling_period_seconds`` defaults to 24 hours.
        """
        if self._execution_rings is None:
            from director_ai.core.execution_rings import ExecutionRingGate

            self._execution_rings = ExecutionRingGate(
                cooling_period_seconds=cooling_period_seconds
            )
        return self._execution_rings

    def output_integrity(
        self, *, signing_seed: bytes | None = None
    ) -> OutputIntegrityGuard:
        """Cryptographic integrity + non-repudiation for model outputs (ML09).

        Signs an output with a detached Ed25519 signature a third party can
        verify with only the public key, and records its digest in an append-only
        tamper-evident
        :class:`~director_ai.core.output_integrity.TamperEvidentLedger`. The
        ledger is stdlib-only and always available; signing needs the optional
        ``cryptography`` backend (``pip install director-ai[crypto]``). Supply a
        32-byte ``signing_seed`` from a secret manager for a stable identity.
        """
        if self._output_integrity is None:
            from director_ai.core.output_integrity import OutputIntegrityGuard

            self._output_integrity = OutputIntegrityGuard(signing_seed=signing_seed)
        return self._output_integrity

    @property
    def ml_bom(self) -> MachineLearningBOM:
        """Supply-chain bill of materials for the ML system (OWASP ASVS).

        Record each model, dataset, and dependency with a SHA-256 digest and
        provenance via
        :class:`~director_ai.core.ml_bom.MachineLearningBOM`, then
        :meth:`~director_ai.core.ml_bom.MachineLearningBOM.verify` the deployed
        artefacts to detect a swapped or poisoned component. The BOM carries its
        own digest so the inventory is itself tamper-evident. Persists on the
        guard so components accumulate across calls.
        """
        if self._ml_bom is None:
            from director_ai.core.ml_bom import MachineLearningBOM

            self._ml_bom = MachineLearningBOM()
        return self._ml_bom

    @property
    def rasp(self) -> RuntimeSelfProtection:
        """Runtime application self-protection from behavioural anomalies.

        The last line of defence once input filters and guardrails are bypassed:
        feed per-request behavioural metrics (request rate, payload size, halt
        rate) to
        :meth:`~director_ai.core.rasp.RuntimeSelfProtection.observe` and read back
        a tenant-safe ok/watch/alert verdict scored by a dependency-free robust
        (median/MAD) detector. Persists across calls so each metric's baseline
        accumulates; the host decides whether to shed load or block.
        """
        if self._rasp is None:
            from director_ai.core.rasp import RuntimeSelfProtection

            self._rasp = RuntimeSelfProtection()
        return self._rasp

    def continuous_fuzzer(self, *, seed: int = 0) -> ContinuousFuzzer:
        """Mutation-based fuzzer that hunts for guard bypasses.

        Where the static adversarial suite checks a fixed list, the returned
        :class:`~director_ai.core.fuzzing.ContinuousFuzzer` mutates a seed corpus
        of attacks (homoglyph, zero-width, leetspeak, delimiter, …) against a
        ``predicate`` you supply — ``True`` = the guard flags it — and reports
        every obfuscation that slipped through plus any seed the guard missed
        outright. The ``seed`` makes a found bypass replayable as a regression.
        """
        from director_ai.core.fuzzing import ContinuousFuzzer

        return ContinuousFuzzer(seed=seed)

    @property
    def threat_intel(self) -> ThreatIntelligenceMatcher:
        """Threat-intelligence IOC matching with attribution (STIX-aligned).

        Register indicators directly or import them from a STIX 2.1 feed with
        :func:`~director_ai.core.threat_intel.from_stix_bundle`, then
        :meth:`~director_ai.core.threat_intel.ThreatIntelligenceMatcher.match`
        prompts/responses against them to get every triggered indicator with its
        attribution and severity — block *and* report "matches the APT29 kit".
        Persists across calls so the indicator set accumulates.
        """
        if self._threat_intel is None:
            from director_ai.core.threat_intel import ThreatIntelligenceMatcher

            self._threat_intel = ThreatIntelligenceMatcher()
        return self._threat_intel

    def robot_command_guard(
        self, constraints: Sequence[PhysicalConstraint] = (), **kwargs: Any
    ) -> RobotCommandGuard:
        """Build an embodied-AI guard for an LLM-planned robot command sequence.

        Verifies a whole plan before execution against per-action physical
        constraints plus temporal caps (bounded step displacement and path
        length). Warn-only by default; pass ``high_risk_enabled=True`` to block an
        unsafe plan. Returns a fresh
        :class:`~director_ai.core.cyber_physical.RobotCommandGuard`; pass
        ``model``, ``high_risk_enabled``, ``max_step_displacement``, or
        ``max_path_length`` via ``kwargs``.
        """
        from director_ai.core.cyber_physical import RobotCommandGuard

        return RobotCommandGuard(constraints, **kwargs)
