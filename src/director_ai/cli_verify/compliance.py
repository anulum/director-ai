# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""director-ai compliance, KPI, forensics and cost-report CLI commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from datetime import UTC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from director_ai.compliance.reporter import Article15TemplateContext


def _load_article15_context(path: str) -> Article15TemplateContext:
    """Load operator-supplied Article 15 context from a JSON file.

    Exits with a clear message when the file is missing, unparsable, or omits a
    required Article 15 narrative field.
    """
    from pathlib import Path

    from director_ai.compliance.reporter import Article15TemplateContext

    ctx_file = Path(path)
    if not ctx_file.exists():
        print(f"Article 15 context file not found: {path}")
        sys.exit(1)
    try:
        data = json.loads(ctx_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid Article 15 context JSON: {exc}")
        sys.exit(1)
    if not isinstance(data, dict):
        print("Article 15 context must be a JSON object")
        sys.exit(1)
    try:
        return Article15TemplateContext(
            system_name=str(data.get("system_name", "")),
            intended_purpose=str(data.get("intended_purpose", "")),
            deployment_context=str(data.get("deployment_context", "")),
            risk_management_summary=str(data.get("risk_management_summary", "")),
            data_governance_summary=str(data.get("data_governance_summary", "")),
            robustness_summary=str(data.get("robustness_summary", "")),
            cybersecurity_summary=str(data.get("cybersecurity_summary", "")),
            human_oversight_summary=str(data.get("human_oversight_summary", "")),
            post_market_monitoring_summary=str(
                data.get("post_market_monitoring_summary", "")
            ),
            known_limitations=tuple(data.get("known_limitations", ())),
            residual_risks=tuple(data.get("residual_risks", ())),
            evidence_refs=tuple(data.get("evidence_refs", ())),
        )
    except ValueError as exc:
        print(f"Incomplete Article 15 context: {exc}")
        sys.exit(1)


def _print_governance(
    *,
    db_path: str,
    fmt: str,
    evidence_root: str,
    config_env: bool,
) -> None:
    """Print the computed governance-controls report.

    A missing audit database is not an error here: the record-keeping
    control degrades to an honest failing signal, which is exactly the
    finding the operator needs to see.
    """
    from pathlib import Path

    from director_ai.compliance.governance_controls import (
        compute_governance_controls,
    )

    audit_log = None
    if Path(db_path).exists():
        from director_ai.compliance.audit_log import AuditLog

        audit_log = AuditLog(db_path)

    config = None
    if config_env:
        from director_ai.core.config import DirectorConfig

        config = DirectorConfig.from_env()

    try:
        report = compute_governance_controls(
            config=config,
            audit_log=audit_log,
            evidence_root=evidence_root,
        )
        if fmt == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.to_markdown())
    finally:
        if audit_log is not None:
            audit_log.close()


def _cmd_compliance(args: list[str]) -> None:
    """EU AI Act Article 15 compliance tools."""
    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage: director-ai compliance <subcommand> [options]\n"
            "\n"
            "Subcommands:\n"
            "  report  [--db PATH] [--since TS] [--until TS]\n"
            "          [--format md|json|html|pdf] [--output FILE.pdf]\n"
            "          [--context FILE.json]  Full Article 15 documentation when an\n"
            "          operator context file is supplied; pdf writes to --output\n"
            "          (default compliance_report.pdf)\n"
            "  status  [--db PATH]   Quick summary\n"
            "  drift   [--db PATH]   Drift detection analysis\n"
            "  anchor  [--db PATH] [--tsa-url URL] [--timeout S]  Anchor the current\n"
            "          audit-chain head with an RFC 3161 timestamp (proves\n"
            "          existence-at-time; TSA from --tsa-url or\n"
            "          DIRECTOR_AUDIT_ANCHOR_TSA_URL)\n"
            "  verify-anchors  [--db PATH]  Verify every stored timestamp anchor\n"
            "          against the audit chain\n"
            "  governance  [--db PATH] [--format md|json] [--config-env]\n"
            "          [--evidence-root DIR]  Computed NIST AI RMF / ISO 42001 /\n"
            "          EU AI Act controls; a missing audit database degrades the\n"
            "          record-keeping signals instead of aborting\n"
            "  evidence-kit  [--db PATH] [--context FILE.json] [--config-env]\n"
            "          [--evidence-root DIR] [--output DIR] [--since TS]\n"
            "          [--until TS]  Write the complete evidence bundle\n"
            "          (governance controls, Article 15 report, SOC 2/ISO\n"
            "          readiness, HIPAA packet, INDEX.md) into one directory\n"
            "          (default compliance_evidence/)\n",
        )
        return

    sub = args[0]
    rest = args[1:]
    db_path = "director_audit.db"
    fmt = "md"
    since = None
    until = None
    context_path = None
    output_path = None
    evidence_root = "."
    config_env = False
    tsa_url = ""
    timeout_s = 10.0

    i = 0
    while i < len(rest):
        if rest[i] == "--db" and i + 1 < len(rest):
            db_path = rest[i + 1]
            i += 2
        elif rest[i] == "--since" and i + 1 < len(rest):
            since = float(rest[i + 1])
            i += 2
        elif rest[i] == "--until" and i + 1 < len(rest):
            until = float(rest[i + 1])
            i += 2
        elif rest[i] == "--format" and i + 1 < len(rest):
            fmt = rest[i + 1]
            i += 2
        elif rest[i] == "--context" and i + 1 < len(rest):
            context_path = rest[i + 1]
            i += 2
        elif rest[i] == "--output" and i + 1 < len(rest):
            output_path = rest[i + 1]
            i += 2
        elif rest[i] == "--evidence-root" and i + 1 < len(rest):
            evidence_root = rest[i + 1]
            i += 2
        elif rest[i] == "--config-env":
            config_env = True
            i += 1
        elif rest[i] == "--tsa-url" and i + 1 < len(rest):
            tsa_url = rest[i + 1]
            i += 2
        elif rest[i] == "--timeout" and i + 1 < len(rest):
            timeout_s = float(rest[i + 1])
            i += 2
        else:
            i += 1

    from pathlib import Path

    if sub == "governance":
        _print_governance(
            db_path=db_path,
            fmt=fmt,
            evidence_root=evidence_root,
            config_env=config_env,
        )
        return

    if sub == "evidence-kit":
        from director_ai.cli_verify.evidence_kit import build_evidence_kit

        context = (
            _load_article15_context(context_path) if context_path is not None else None
        )
        config = None
        if config_env:
            from director_ai.core.config import DirectorConfig

            config = DirectorConfig.from_env()
        kit = build_evidence_kit(
            db_path=db_path,
            context=context,
            config=config,
            evidence_root=evidence_root,
            out_dir=output_path or "compliance_evidence",
            since=since,
            until=until,
        )
        for section in kit.sections:
            marker = "+" if section.included else "-"
            print(f"  {marker} {section.name}: {section.note}")
        print(kit.summary_line())
        return

    if not Path(db_path).exists():
        print(f"Audit database not found: {db_path}")
        print(
            "Run the proxy or server with --audit-db / DIRECTOR_COMPLIANCE_DB_PATH first."
        )
        sys.exit(1)

    from director_ai.compliance.audit_log import AuditLog
    from director_ai.compliance.drift_detector import DriftDetector
    from director_ai.compliance.reporter import ComplianceReporter

    log = AuditLog(db_path)

    if sub == "report":
        reporter = ComplianceReporter(log)
        report = reporter.generate_report(since=since, until=until)
        if context_path is not None:
            # Full Article 15 technical documentation: the operator-authored
            # context (system purpose, risk/data-governance/oversight summaries)
            # cannot be derived from telemetry, so it is supplied as a file. With
            # it, emit the complete regulator-grade record rather than a summary.
            context = _load_article15_context(context_path)
            if fmt == "json":
                print(json.dumps(report.to_article15_template(context), indent=2))
            else:
                print(report.to_article15_markdown(context))
        elif fmt == "json":
            print(
                json.dumps(
                    {
                        "total_interactions": report.total_interactions,
                        "hallucination_rate": report.overall_hallucination_rate,
                        "hallucination_rate_ci": report.overall_hallucination_rate_ci,
                        "avg_score": report.avg_score,
                        "drift_detected": report.drift_detected,
                        "incident_count": report.incident_count,
                    },
                    indent=2,
                )
            )
        elif fmt in ("html", "pdf"):
            from director_ai.compliance.report_templates import (
                render_compliance_html,
                render_compliance_pdf,
            )

            report_data = {
                "title": "EU AI Act Article 15 Report",
                "period": f"{since or 'start'} to {until or 'now'}",
                "hallucination_rate": report.overall_hallucination_rate,
                "total_reviews": report.total_interactions,
                "approved_count": getattr(report, "approved_count", 0),
                "rejected_count": getattr(report, "rejected_count", 0),
                "avg_score": report.avg_score,
                "avg_latency_ms": getattr(report, "avg_latency_ms", 0),
                "drift_detected": report.drift_detected,
                "models": getattr(report, "model_breakdown", []),
            }
            if fmt == "html":
                print(render_compliance_html(report_data))
            else:
                out = output_path or "compliance_report.pdf"
                with open(out, "wb") as fh:
                    fh.write(render_compliance_pdf(report_data))
                print(f"Wrote PDF compliance report to {out}")
        else:
            print(report.to_markdown())

    elif sub == "status":
        reporter = ComplianceReporter(log)
        report = reporter.generate_report(since=since, until=until)
        n = report.total_interactions
        hr = report.overall_hallucination_rate
        drift = "YES" if report.drift_detected else "no"
        print(
            f"Interactions: {n:,} | "
            f"Hallucination rate: {hr:.2%} | "
            f"Incidents: {report.incident_count:,} | "
            f"Drift: {drift}"
        )

    elif sub == "drift":
        detector = DriftDetector(log)
        result = detector.analyze(since=since, until=until)
        status = f"DETECTED ({result.severity})" if result.detected else "none"
        print(
            f"Drift: {status} | "
            f"z={result.z_score:.2f} p={result.p_value:.4f} | "
            f"Rate change: {result.rate_change:+.2%} | "
            f"Windows: {len(result.windows)}"
        )

    elif sub == "anchor":
        import os
        from datetime import datetime

        from director_ai.compliance.timestamp_anchor import (
            AnchorStore,
            Rfc3161Anchorer,
            try_anchor_chain_head,
        )

        url = tsa_url or os.environ.get("DIRECTOR_AUDIT_ANCHOR_TSA_URL", "")
        if not url:
            print(
                "No TSA URL — pass --tsa-url URL or set DIRECTOR_AUDIT_ANCHOR_TSA_URL."
            )
            log.close()
            sys.exit(1)
        head = log.current_head()
        store = AnchorStore(db_path)
        try:
            anchor = try_anchor_chain_head(
                head, Rfc3161Anchorer(url, timeout_s=timeout_s), store
            )
        finally:
            store.close()
        if anchor is None:
            if head == "0" * 64:
                print("Audit chain is empty — nothing to anchor.")
            else:
                print(f"Anchoring failed (TSA unreachable or rejected): {url}")
            log.close()
            sys.exit(1)
        when = datetime.fromtimestamp(anchor.gen_time, tz=UTC).isoformat()
        print(
            f"Anchored chain head {anchor.anchored_hash[:16]}… at {when} "
            f"(TSA serial {anchor.serial_number}, {url})"
        )

    elif sub == "verify-anchors":
        from director_ai.compliance.timestamp_anchor import AnchorStore

        store = AnchorStore(db_path)
        try:
            count = len(store.all())
            ok, bad = store.verify_against_chain()
        finally:
            store.close()
        if count == 0:
            print("No timestamp anchors recorded yet.")
        elif ok:
            print(f"All {count} timestamp anchor(s) verify against the audit chain.")
        else:
            print(f"Anchor verification FAILED at head {bad}")
            log.close()
            sys.exit(1)

    else:
        print(f"Unknown compliance subcommand: {sub}")
        sys.exit(1)

    log.close()


def _cmd_cost_report(args: list[str]) -> None:
    """Show token cost report from the running scorer's CostAnalyser."""
    if _is_help_request(args):
        _print_cost_report_help()
        return

    fmt = "text"
    i = 0
    while i < len(args):
        if args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            i += 2
        else:
            i += 1

    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    if not cfg.cost_tracking_enabled:
        print(
            "Cost tracking is disabled. "
            "Set DIRECTOR_COST_TRACKING_ENABLED=true or cost_tracking_enabled: true"
        )
        sys.exit(1)

    scorer = cfg.build_scorer()
    analyser = getattr(scorer, "_cost_analyser", None)
    if analyser is None:
        print("No CostAnalyser attached to scorer.")
        sys.exit(1)

    report = analyser.report()

    if fmt == "json":
        print(json.dumps(report, indent=2))
    elif fmt == "html":
        from director_ai.compliance.report_templates import render_cost_html

        print(render_cost_html(report))
    else:
        print(f"Total cost: {report['currency']} {report['total_cost']:.6f}")
        print(f"Total tokens: {report['total_tokens']:,}")
        for key, m in report.get("models", {}).items():
            print(
                f"  {key}: {m['call_count']} calls, "
                f"{m['total_tokens']:,} tokens, "
                f"{report['currency']} {m['estimated_cost']:.6f}"
            )


def _kpi_items_from_records(records: list[Any]) -> list[Any]:
    """Build :class:`LabelItem`s from JSON records (only the fields it accepts)."""
    from director_ai.core.labelling_cockpit.items import LabelItem

    fields = ("item_id", "score", "guard_approved", "domain", "label")
    items = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each item must be a JSON object")
        kwargs = {key: record[key] for key in fields if key in record}
        items.append(LabelItem(**kwargs))
    return items


def _cmd_kpis(args: list[str]) -> None:
    """Render the board-level guardrail KPIs from a JSON input bundle.

    The input file is a JSON object with ``items`` (reviewer-labelled guard
    decisions), optional ``latency_ms_samples``, optional operational counters
    (``tenant_boundary_violations`` / ``unsigned_kb_writes_rejected`` /
    ``security_exception_debt``), and an optional ``targets`` overlay.
    """
    if _is_help_request(args):
        _print_kpis_help()
        return

    input_path = None
    fmt = "text"
    i = 0
    while i < len(args):
        if args[i] == "--input" and i + 1 < len(args):
            input_path = args[i + 1]
            i += 2
        elif args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            i += 2
        else:
            i += 1

    if input_path is None:
        _print_kpis_help()
        sys.exit(1)
    if fmt not in ("text", "markdown", "json"):
        print(f"Unknown format '{fmt}'. Choose from: text, markdown, json")
        sys.exit(1)

    from pathlib import Path

    bundle_path = Path(input_path)
    if not bundle_path.exists():
        print(f"Error: input bundle not found at {bundle_path}")
        sys.exit(1)

    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {bundle_path}: {exc}")
        sys.exit(1)
    if not isinstance(bundle, Mapping):
        print("Error: input bundle must be a JSON object")
        sys.exit(1)

    from director_ai.core.observability.kpi_report import (
        KpiTargets,
        render_markdown,
        render_text,
    )
    from director_ai.core.observability.kpis import compute_kpis

    try:
        items = _kpi_items_from_records(list(bundle.get("items", [])))
    except (ValueError, TypeError) as exc:
        print(f"Error: invalid item record: {exc}")
        sys.exit(1)

    report = compute_kpis(
        items,
        latency_ms_samples=list(bundle.get("latency_ms_samples", [])),
        tenant_boundary_violations=int(bundle.get("tenant_boundary_violations", 0)),
        unsigned_kb_writes_rejected=int(bundle.get("unsigned_kb_writes_rejected", 0)),
        security_exception_debt=int(bundle.get("security_exception_debt", 0)),
    )

    targets_overlay = bundle.get("targets")
    try:
        targets = KpiTargets(**targets_overlay) if targets_overlay else None
    except (ValueError, TypeError) as exc:
        print(f"Error: invalid targets overlay: {exc}")
        sys.exit(1)

    if fmt == "json":
        from director_ai.core.observability.kpi_report import (
            kpi_statuses,
            overall_status,
        )

        payload = {
            "report": report.to_dict(),
            "statuses": kpi_statuses(report, targets),
            "overall": overall_status(report, targets),
        }
        print(json.dumps(payload, indent=2))
    elif fmt == "markdown":
        print(render_markdown(report, targets=targets))
    else:
        print(render_text(report, targets=targets))


def _cmd_forensics(args: list[str]) -> None:
    """Render scorer-miss forensics from tenant-safe eval records."""
    if _is_help_request(args):
        _print_forensics_help()
        return

    input_path = None
    fmt = "text"
    i = 0
    while i < len(args):
        if args[i] == "--input" and i + 1 < len(args):
            input_path = args[i + 1]
            i += 2
        elif args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            i += 2
        else:
            i += 1

    if input_path is None:
        _print_forensics_help()
        sys.exit(1)
    if fmt not in ("text", "markdown", "json"):
        print(f"Unknown format '{fmt}'. Choose from: text, markdown, json")
        sys.exit(1)

    from pathlib import Path

    records_path = Path(input_path)
    if not records_path.exists():
        print(f"Error: input records not found at {records_path}")
        sys.exit(1)

    try:
        payload = json.loads(records_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {records_path}: {exc}")
        sys.exit(1)

    try:
        records = _forensics_records_from_payload(payload)
    except ValueError as exc:
        print(f"Error: invalid forensics input: {exc}")
        sys.exit(1)

    from director_ai.core.observability.forensics import (
        build_forensics_report,
        render_forensics_markdown,
        render_forensics_text,
    )

    try:
        report = build_forensics_report(records)
    except (TypeError, ValueError) as exc:
        print(f"Error: invalid forensics record: {exc}")
        sys.exit(1)

    if fmt == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif fmt == "markdown":
        print(render_forensics_markdown(report))
    else:
        print(render_forensics_text(report))


def _forensics_records_from_payload(payload: object) -> list[Mapping[str, object]]:
    """Return record mappings from either a list or ``{"records": [...]}``."""
    records_obj: object
    if isinstance(payload, Mapping):
        records_obj = payload.get("records", [])
    else:
        records_obj = payload
    if not isinstance(records_obj, list):
        raise ValueError("records must be a JSON array")
    records: list[Mapping[str, object]] = []
    for record in records_obj:
        if not isinstance(record, Mapping):
            raise ValueError("each record must be a JSON object")
        records.append(record)
    return records


def _is_help_request(args: list[str]) -> bool:
    """Return whether a command received an explicit help token."""
    return bool(args) and args[0] in ("-h", "--help", "help")


def _print_cost_report_help() -> None:
    """Print cost-report usage without loading scorer configuration."""
    print(
        "Usage: director-ai cost-report [--format text|json|html]\n"
        "\n"
        "Render the running scorer cost tracking report when cost tracking is enabled.\n"
    )


def _print_kpis_help() -> None:
    """Print KPI usage without opening the input bundle."""
    print(
        "Usage: director-ai kpis --input <bundle.json> "
        "[--format text|markdown|json]\n"
        "\n"
        "Render board-level guardrail KPIs from a JSON input bundle.\n"
    )


def _print_forensics_help() -> None:
    """Print forensics usage without opening eval records."""
    print(
        "Usage: director-ai forensics --input <records.json> "
        "[--format text|markdown|json]\n"
        "\n"
        "Render scorer-miss forensics from tenant-safe eval records.\n"
    )
