# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — computed AI-governance controls (NIST AI RMF / ISO 42001 / EU AI Act)

"""Computed AI-governance controls with a NIST / ISO / EU AI Act crosswalk.

Unlike the static SOC 2 / ISO 27001 readiness catalogue in
:mod:`director_ai.compliance.readiness`, every control here derives its
status from **observable deployment state** at call time: DirectorConfig
knobs, the attached tamper-evident audit log (including a live
``verify_chain()`` pass), and the presence of evidence artefacts under
the operator-supplied evidence root. Each control carries crosswalk
references to NIST AI RMF 1.0 functions (category level), ISO/IEC
42001:2023 clauses (clause / Annex A level — deliberately coarse), and
EU AI Act articles.

The output is readiness evidence for operator review. It is not a
certification, a conformity assessment, an audit opinion, or legal
advice, and it never serialises raw prompts, audit rows, or secrets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .readiness import ReadinessStatus, _risk_level

if TYPE_CHECKING:
    from director_ai.core.config import DirectorConfig

    from .audit_log import AuditLog

_NIST_FUNCTIONS = ("GOVERN", "MAP", "MEASURE", "MANAGE")
_ISO42001_PREFIXES = ("Clause ", "A.")
_EU_AI_ACT_PREFIX = "Article "

_DISCLAIMER = (
    "Computed governance-readiness evidence only; this is not an EU AI Act "
    "conformity assessment, a NIST AI RMF attestation, an ISO/IEC 42001 "
    "certification, an audit opinion, or legal advice."
)


@dataclass(frozen=True)
class ControlSignal:
    """One observable signal contributing to a computed control.

    Parameters
    ----------
    name:
        Stable snake_case signal identifier.
    observed:
        Tenant-safe, human-readable statement of what was observed at
        computation time. Never contains raw prompts, rows, or secrets.
    satisfied:
        Whether the observation satisfies the control expectation.
    """

    name: str
    observed: str
    satisfied: bool

    def __post_init__(self) -> None:
        """Validate the signal identifier and observation text."""
        if not self.name.strip():
            raise ValueError("signal name is required")
        if not self.observed.strip():
            raise ValueError("signal observation is required")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "observed", self.observed.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return the tenant-safe JSON shape of the signal."""
        return {
            "name": self.name,
            "observed": self.observed,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True)
class GovernanceControl:
    """One computed control with a NIST / ISO 42001 / EU AI Act crosswalk.

    Parameters
    ----------
    control_id:
        Stable uppercase identifier used in reports.
    title:
        Human-readable control statement.
    nist_ai_rmf_refs:
        NIST AI RMF 1.0 references at function/category level, such as
        ``GOVERN 1`` or ``MEASURE 2``.
    iso42001_refs:
        ISO/IEC 42001:2023 references at clause or Annex A level, such
        as ``Clause 6.1`` or ``A.7``.
    eu_ai_act_refs:
        EU AI Act references such as ``Article 9(2)``.
    signals:
        Observable signals the status is derived from.

    Notes
    -----
    The status is **derived**, never stored: every signal satisfied →
    ``passed``; at least one satisfied → ``warning``; none satisfied →
    ``failing``; no signals → ``not_applicable``.
    """

    control_id: str
    title: str
    nist_ai_rmf_refs: tuple[str, ...]
    iso42001_refs: tuple[str, ...]
    eu_ai_act_refs: tuple[str, ...]
    signals: tuple[ControlSignal, ...]

    def __post_init__(self) -> None:
        """Normalise the id and validate every crosswalk reference."""
        control_id = self.control_id.strip().upper()
        if not control_id or not control_id.replace("-", "").isalnum():
            raise ValueError("control_id must contain letters, numbers, or hyphen")
        if not self.title.strip():
            raise ValueError("title is required")

        nist = tuple(item.strip() for item in self.nist_ai_rmf_refs if item)
        if not nist or any(not ref.startswith(_NIST_FUNCTIONS) for ref in nist):
            raise ValueError(
                f"nist_ai_rmf_refs must start with one of {', '.join(_NIST_FUNCTIONS)}",
            )

        iso = tuple(item.strip() for item in self.iso42001_refs if item)
        if not iso or any(not ref.startswith(_ISO42001_PREFIXES) for ref in iso):
            raise ValueError(
                "iso42001_refs must use clause or Annex A references such "
                "as 'Clause 6.1' or 'A.7'",
            )

        eu = tuple(item.strip() for item in self.eu_ai_act_refs if item)
        if not eu or any(not ref.startswith(_EU_AI_ACT_PREFIX) for ref in eu):
            raise ValueError(
                "eu_ai_act_refs must use references such as 'Article 9(2)'",
            )

        object.__setattr__(self, "control_id", control_id)
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "nist_ai_rmf_refs", nist)
        object.__setattr__(self, "iso42001_refs", iso)
        object.__setattr__(self, "eu_ai_act_refs", eu)
        object.__setattr__(self, "signals", tuple(self.signals))

    @property
    def status(self) -> ReadinessStatus:
        """Derive the control status from its signals."""
        if not self.signals:
            return ReadinessStatus.NOT_APPLICABLE
        satisfied = sum(1 for signal in self.signals if signal.satisfied)
        if satisfied == len(self.signals):
            return ReadinessStatus.PASS
        if satisfied:
            return ReadinessStatus.WARNING
        return ReadinessStatus.FAIL

    def to_dict(self) -> dict[str, Any]:
        """Return tenant-safe JSON-compatible control metadata."""
        return {
            "control_id": self.control_id,
            "title": self.title,
            "nist_ai_rmf_refs": list(self.nist_ai_rmf_refs),
            "iso42001_refs": list(self.iso42001_refs),
            "eu_ai_act_refs": list(self.eu_ai_act_refs),
            "status": self.status.value,
            "signals": [signal.to_dict() for signal in self.signals],
        }


@dataclass(frozen=True)
class GovernanceControlsReport:
    """Tenant-safe computed AI-governance controls report.

    Parameters
    ----------
    generated_at:
        UTC timestamp for the report payload.
    controls:
        Computed controls included in the report.
    inputs:
        Tenant-safe description of the inputs the computation saw
        (whether a config and an audit log were attached, and the
        evidence root that was scanned).
    """

    generated_at: str
    controls: tuple[GovernanceControl, ...]
    inputs: dict[str, Any]

    def summary(self) -> dict[str, int | float | str]:
        """Return aggregate control counts and risk level."""
        total = len(self.controls)
        passed = sum(
            1 for control in self.controls if control.status is ReadinessStatus.PASS
        )
        warnings = sum(
            1 for control in self.controls if control.status is ReadinessStatus.WARNING
        )
        failures = sum(
            1 for control in self.controls if control.status is ReadinessStatus.FAIL
        )
        not_applicable = sum(
            1
            for control in self.controls
            if control.status is ReadinessStatus.NOT_APPLICABLE
        )
        applicable = max(total - not_applicable, 1)
        return {
            "total_controls": total,
            "passed": passed,
            "warnings": warnings,
            "failures": failures,
            "not_applicable": not_applicable,
            "readiness_score": round(passed / applicable, 4),
            "risk_level": _risk_level(failures=failures, warnings=warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a tenant-safe JSON-compatible governance payload."""
        return {
            "frameworks": [
                "NIST AI RMF 1.0",
                "ISO/IEC 42001:2023",
                "EU AI Act (Regulation (EU) 2024/1689)",
            ],
            "generated_at": self.generated_at,
            "computed": True,
            "inputs": dict(self.inputs),
            "summary": self.summary(),
            "controls": [control.to_dict() for control in self.controls],
            "disclaimer": _DISCLAIMER,
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_interaction_text_included": False,
                "raw_security_evidence_included": False,
                "certification_claimed": False,
            },
        }

    def to_markdown(self) -> str:
        """Render the computed controls report as Markdown."""
        lines = [
            "# Computed AI-Governance Controls",
            "",
            f"Generated: {self.generated_at}",
            "",
            "## Summary",
        ]
        for key, value in self.summary().items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        lines.extend(
            [
                "",
                "## Controls",
                "",
                "| ID | Control | Status | NIST AI RMF | ISO/IEC 42001 | EU AI Act | Signals |",
                "|---|---|---:|---|---|---|---|",
            ]
        )
        for control in self.controls:
            signals = "; ".join(
                f"{'✓' if signal.satisfied else '✗'} {signal.name}"
                for signal in control.signals
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        control.control_id,
                        control.title,
                        control.status.value,
                        ", ".join(control.nist_ai_rmf_refs),
                        ", ".join(control.iso42001_refs),
                        ", ".join(control.eu_ai_act_refs),
                        signals,
                    ]
                )
                + " |"
            )
        lines.extend(["", f"> {_DISCLAIMER}"])
        return "\n".join(lines)


def _config_signal(
    config: DirectorConfig | None,
    name: str,
    *,
    absent: str,
    present: str,
    satisfied: bool,
) -> ControlSignal:
    """Build a config-derived signal that degrades honestly without config."""
    if config is None:
        return ControlSignal(name=name, observed=absent, satisfied=False)
    return ControlSignal(name=name, observed=present, satisfied=satisfied)


def _risk_management_control(
    config: DirectorConfig | None,
    audit_log: AuditLog | None,
) -> GovernanceControl:
    """Article 9 risk-management control computed from live guard knobs."""
    signals = [
        _config_signal(
            config,
            "guard_thresholds_configured",
            absent="no DirectorConfig supplied",
            present=(
                f"hard_limit={getattr(config, 'hard_limit', None)}, "
                f"coherence_threshold={getattr(config, 'coherence_threshold', None)}"
            ),
            satisfied=bool(
                config is not None
                and 0.0 < config.hard_limit <= config.coherence_threshold < 1.0
            ),
        ),
        _config_signal(
            config,
            "adaptive_threshold_governor",
            absent="no DirectorConfig supplied",
            present=(
                "adaptive_threshold_enabled="
                f"{getattr(config, 'adaptive_threshold_enabled', None)}"
            ),
            satisfied=bool(config is not None and config.adaptive_threshold_enabled),
        ),
        ControlSignal(
            name="drift_monitoring_wired",
            observed=(
                "audit log attached; drift analysis runs over recorded periods"
                if audit_log is not None
                else "no audit log attached; drift analysis has no data source"
            ),
            satisfied=audit_log is not None,
        ),
    ]
    return GovernanceControl(
        control_id="GOV-RISK-01",
        title="Risk management thresholds, adaptive governor, and drift monitoring",
        nist_ai_rmf_refs=("GOVERN 1", "MANAGE 1"),
        iso42001_refs=("Clause 6.1", "Clause 8.2"),
        eu_ai_act_refs=("Article 9(1)", "Article 9(2)"),
        signals=tuple(signals),
    )


def _data_governance_control(config: DirectorConfig | None) -> GovernanceControl:
    """Article 10 data-governance control computed from grounding knobs."""
    signals = [
        _config_signal(
            config,
            "grounding_store_configured",
            absent="no DirectorConfig supplied",
            present=f"vector_backend={getattr(config, 'vector_backend', '')!r}",
            satisfied=bool(config is not None and config.vector_backend),
        ),
        _config_signal(
            config,
            "pii_redaction_enabled",
            absent="no DirectorConfig supplied",
            present=f"redact_pii={getattr(config, 'redact_pii', None)}",
            satisfied=bool(config is not None and config.redact_pii),
        ),
        _config_signal(
            config,
            "tenant_isolation_enabled",
            absent="no DirectorConfig supplied",
            present=f"tenant_routing={getattr(config, 'tenant_routing', None)}",
            satisfied=bool(config is not None and config.tenant_routing),
        ),
    ]
    return GovernanceControl(
        control_id="GOV-DATA-01",
        title="Grounding-corpus governance, PII redaction, and tenant isolation",
        nist_ai_rmf_refs=("MAP 2", "MEASURE 2"),
        iso42001_refs=("A.7",),
        eu_ai_act_refs=("Article 10(2)", "Article 10(5)"),
        signals=tuple(signals),
    )


def _technical_documentation_control(
    audit_log: AuditLog | None,
    evidence_root: Path,
) -> GovernanceControl:
    """Article 11 documentation control computed from evidence artefacts."""
    manifest = evidence_root / "docs" / "_generated" / "capability_manifest.json"
    public_benchmarks = evidence_root / "benchmarks" / "PUBLIC_BENCHMARKS.md"
    signals = [
        ControlSignal(
            name="capability_inventory_present",
            observed=(
                f"{manifest} exists"
                if manifest.is_file()
                else f"{manifest} not found under evidence root"
            ),
            satisfied=manifest.is_file(),
        ),
        ControlSignal(
            name="public_benchmark_evidence_present",
            observed=(
                f"{public_benchmarks} exists"
                if public_benchmarks.is_file()
                else f"{public_benchmarks} not found under evidence root"
            ),
            satisfied=public_benchmarks.is_file(),
        ),
        ControlSignal(
            name="article15_reporting_wired",
            observed=(
                "audit log attached; Article 15 reports can be generated"
                if audit_log is not None
                else "no audit log attached; Article 15 reports unavailable"
            ),
            satisfied=audit_log is not None,
        ),
    ]
    return GovernanceControl(
        control_id="GOV-DOC-01",
        title="Technical documentation and capability inventory evidence",
        nist_ai_rmf_refs=("GOVERN 4",),
        iso42001_refs=("Clause 7.5",),
        eu_ai_act_refs=("Article 11(1)",),
        signals=tuple(signals),
    )


def _record_keeping_control(audit_log: AuditLog | None) -> GovernanceControl:
    """Article 12 record-keeping control with a live chain verification."""
    if audit_log is None:
        chain_signal = ControlSignal(
            name="tamper_evident_chain_verified",
            observed="no audit log attached; chain verification not possible",
            satisfied=False,
        )
        entries_signal = ControlSignal(
            name="audit_entries_recorded",
            observed="no audit log attached",
            satisfied=False,
        )
    else:
        chain_ok, first_bad = audit_log.verify_chain()
        chain_signal = ControlSignal(
            name="tamper_evident_chain_verified",
            observed=(
                "hash chain re-derived and intact"
                if chain_ok
                else f"chain verification FAILED at row {first_bad}"
            ),
            satisfied=chain_ok,
        )
        entry_count = audit_log.count()
        entries_signal = ControlSignal(
            name="audit_entries_recorded",
            observed=f"{entry_count} sealed entries",
            satisfied=entry_count > 0,
        )
    signals = [
        ControlSignal(
            name="audit_log_attached",
            observed=(
                "tamper-evident audit log attached"
                if audit_log is not None
                else "no audit log attached"
            ),
            satisfied=audit_log is not None,
        ),
        chain_signal,
        entries_signal,
    ]
    return GovernanceControl(
        control_id="GOV-LOG-01",
        title="Record-keeping with tamper-evident, verifiable audit chain",
        nist_ai_rmf_refs=("MANAGE 4",),
        iso42001_refs=("Clause 9.1",),
        eu_ai_act_refs=("Article 12(1)", "Article 12(2)"),
        signals=tuple(signals),
    )


def _accuracy_monitoring_control(audit_log: AuditLog | None) -> GovernanceControl:
    """Article 15 bridge: accuracy metrics computed from recorded traffic."""
    entry_count = audit_log.count() if audit_log is not None else 0
    signals = [
        ControlSignal(
            name="accuracy_reporting_available",
            observed=(
                "audit log attached; FPR/FNR, hallucination-rate, and drift "
                "reports can be generated"
                if audit_log is not None
                else "no audit log attached; accuracy reporting unavailable"
            ),
            satisfied=audit_log is not None,
        ),
        ControlSignal(
            name="interactions_recorded_for_metrics",
            observed=f"{entry_count} recorded interactions",
            satisfied=entry_count > 0,
        ),
    ]
    return GovernanceControl(
        control_id="GOV-ACC-01",
        title="Continuous accuracy monitoring feeding Article 15 reports",
        nist_ai_rmf_refs=("MEASURE 2",),
        iso42001_refs=("Clause 9.1",),
        eu_ai_act_refs=("Article 15(1)", "Article 15(4)"),
        signals=tuple(signals),
    )


def _human_oversight_control(
    config: DirectorConfig | None,
    audit_log: AuditLog | None,
) -> GovernanceControl:
    """Article 14 oversight control computed from review-band and audit knobs."""
    signals = [
        _config_signal(
            config,
            "review_band_configured",
            absent="no DirectorConfig supplied",
            present=(
                f"hard_limit={getattr(config, 'hard_limit', None)} < "
                f"coherence_threshold={getattr(config, 'coherence_threshold', None)} "
                "leaves a human-review band"
            ),
            satisfied=bool(
                config is not None and config.hard_limit < config.coherence_threshold
            ),
        ),
        ControlSignal(
            name="override_tracking_available",
            observed=(
                "audit log attached; human override rate is computable"
                if audit_log is not None
                else "no audit log attached; overrides are not recorded"
            ),
            satisfied=audit_log is not None,
        ),
    ]
    return GovernanceControl(
        control_id="GOV-OVR-01",
        title="Human-oversight review band and override tracking",
        nist_ai_rmf_refs=("GOVERN 3", "MANAGE 2"),
        iso42001_refs=("Clause 5.3",),
        eu_ai_act_refs=("Article 14(1)", "Article 14(4)"),
        signals=tuple(signals),
    )


def compute_governance_controls(
    *,
    config: DirectorConfig | None = None,
    audit_log: AuditLog | None = None,
    evidence_root: str | Path = ".",
    generated_at: str = "",
) -> GovernanceControlsReport:
    """Compute the AI-governance controls report from live deployment state.

    Parameters
    ----------
    config:
        The deployment DirectorConfig. ``None`` degrades every
        config-derived signal to unsatisfied with an honest observation.
    audit_log:
        The tamper-evident compliance audit log. When attached, the
        record-keeping control re-derives the hash chain live.
    evidence_root:
        Directory scanned for documentation evidence artefacts
        (capability inventory, public benchmark index).
    generated_at:
        Optional UTC timestamp. If omitted, the current UTC time is used.

    Returns
    -------
    GovernanceControlsReport
        Tenant-safe computed controls with the framework crosswalk.
    """
    root = Path(evidence_root)
    controls = (
        _risk_management_control(config, audit_log),
        _data_governance_control(config),
        _technical_documentation_control(audit_log, root),
        _record_keeping_control(audit_log),
        _accuracy_monitoring_control(audit_log),
        _human_oversight_control(config, audit_log),
    )
    return GovernanceControlsReport(
        generated_at=generated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        controls=controls,
        inputs={
            "config_attached": config is not None,
            "audit_log_attached": audit_log is not None,
            "evidence_root": str(root),
        },
    )
