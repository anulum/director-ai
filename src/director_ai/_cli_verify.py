# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI Verification/Diagnostics Commands
"""CLI subcommands for verification, diagnostics, compliance, and licensing.

Extracted from cli.py to reduce module size.
"""

from __future__ import annotations

import json
import sys


def _cmd_doctor(args: list[str]) -> None:
    """Check runtime dependencies and print readiness summary."""
    import platform

    import director_ai

    checks: list[tuple[str, bool, str]] = []

    # Python version
    py_ver = platform.python_version()
    py_ok = tuple(int(x) for x in py_ver.split(".")[:2]) >= (3, 11)
    checks.append(("Python >= 3.11", py_ok, py_ver))

    # torch
    try:
        import torch

        cuda = torch.cuda.is_available()
        checks.append(("torch", True, f"{torch.__version__} (CUDA: {cuda})"))
    except ImportError:
        checks.append(("torch", False, "not installed"))

    # transformers
    try:
        import transformers

        checks.append(("transformers", True, transformers.__version__))
    except ImportError:
        checks.append(("transformers", False, "not installed"))

    # NLI model availability
    try:
        from director_ai.core.scoring.nli import nli_available

        avail = nli_available()
        detail = "torch+transformers" if avail else "missing deps"
        checks.append(("NLI model ready", avail, detail))
    except Exception as exc:
        checks.append(("NLI model ready", False, str(exc)))

    # onnxruntime
    try:
        import onnxruntime as ort

        provs = ort.get_available_providers()
        checks.append(("onnxruntime", True, f"{ort.__version__} ({', '.join(provs)})"))
    except ImportError:
        checks.append(("onnxruntime", False, "not installed"))

    # chromadb
    try:
        import chromadb

        checks.append(("chromadb", True, chromadb.__version__))
    except ImportError:
        checks.append(("chromadb", False, "not installed"))

    # sentence_transformers
    try:
        import sentence_transformers

        ver = sentence_transformers.__version__
        checks.append(("sentence-transformers", True, ver))
    except ImportError:
        checks.append(("sentence-transformers", False, "not installed"))

    # slowapi
    try:
        import slowapi

        checks.append(("slowapi", True, getattr(slowapi, "__version__", "installed")))
    except ImportError:
        checks.append(("slowapi", False, "not installed"))

    # grpcio
    try:
        import grpc

        checks.append(("grpcio", True, grpc.__version__))
    except ImportError:
        checks.append(("grpcio", False, "not installed"))

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)

    print(f"director-ai {director_ai.__version__} — dependency check\n")
    for name, ok, detail in checks:
        mark = "+" if ok else "-"
        print(f"  [{mark}] {name}: {detail}")
    print(f"\n{passed}/{total} checks passed")


def _cmd_license(args: list[str]) -> None:
    """License management: status, generate, validate."""
    from .core.license import generate_license, load_license, validate_file

    if not args or args[0] == "status":
        info = load_license()
        print(f"Tier:     {info.tier}")
        print(f"Valid:    {info.valid}")
        print(f"Licensee: {info.licensee or '(community)'}")
        if info.expires:
            print(f"Expires:  {info.expires}")
        if info.key:
            print(f"Key:      {info.key[:20]}...")
        print(f"Message:  {info.message}")
        return

    if args[0] == "generate":
        import os

        admin_key = os.environ.get("DIRECTOR_ADMIN_KEY", "")
        if not admin_key:
            print(
                "Error: DIRECTOR_ADMIN_KEY environment variable required for license generation."
            )
            print("This command is for license administrators only.")
            sys.exit(1)

        import argparse

        p = argparse.ArgumentParser(prog="director-ai license generate")
        p.add_argument(
            "--tier", required=True, choices=["indie", "pro", "enterprise", "trial"]
        )
        p.add_argument("--licensee", required=True)
        p.add_argument("--email", required=True)
        p.add_argument("--days", type=int, default=365)
        p.add_argument("--deployments", type=int, default=1)
        p.add_argument("--output", default="license.json")
        parsed = p.parse_args(args[1:])

        import json
        from pathlib import Path

        data = generate_license(
            tier=parsed.tier,
            licensee=parsed.licensee,
            email=parsed.email,
            days=parsed.days,
            deployments=parsed.deployments,
        )
        Path(parsed.output).write_text(json.dumps(data, indent=2) + "\n")
        print(f"License generated: {parsed.output}")
        print(f"Key: {data['key']}")
        print(f"Tier: {data['tier']}")
        print(f"Licensee: {data['licensee']}")
        print(f"Expires: {data['expires']}")
        return

    if args[0] == "validate":
        if len(args) < 2:
            print("Usage: director-ai license validate <path>")
            sys.exit(1)
        info = validate_file(args[1])
        print(f"Valid:    {info.valid}")
        print(f"Tier:     {info.tier}")
        print(f"Licensee: {info.licensee}")
        print(f"Message:  {info.message}")
        sys.exit(0 if info.valid else 1)

    print(f"Unknown license subcommand: {args[0]}")
    print("Usage: director-ai license [status|generate|validate]")
    sys.exit(1)


def _cmd_compliance(args: list[str]) -> None:
    """EU AI Act Article 15 compliance tools."""
    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage: director-ai compliance <subcommand> [options]\n"
            "\n"
            "Subcommands:\n"
            "  report  [--db PATH] [--since TS] [--until TS] [--format md|json|html]\n"
            "  status  [--db PATH]   Quick summary\n"
            "  drift   [--db PATH]   Drift detection analysis\n",
        )
        return

    sub = args[0]
    rest = args[1:]
    db_path = "director_audit.db"
    fmt = "md"
    since = None
    until = None

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
        else:
            i += 1

    from pathlib import Path

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
        if fmt == "json":
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
        elif fmt == "html":
            from director_ai.compliance.report_templates import render_compliance_html

            html = render_compliance_html(
                {
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
            )
            print(html)
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

    else:
        print(f"Unknown compliance subcommand: {sub}")
        sys.exit(1)

    log.close()


def _cmd_verify_numeric(args: list[str]) -> None:
    """Check numeric consistency in text."""
    if not args:
        print("Usage: director-ai verify-numeric <text>")
        sys.exit(1)

    from director_ai.core.verification.numeric_verifier import verify_numeric

    result = verify_numeric(" ".join(args))
    print(f"Valid:    {result.valid}")
    print(f"Claims:  {result.claims_found}")
    print(f"Errors:  {result.error_count}")
    print(f"Warnings:{result.warning_count}")
    for issue in result.issues:
        print(f"  [{issue.severity}] {issue.issue_type}: {issue.description}")


def _cmd_verify_reasoning(args: list[str]) -> None:
    """Verify logical structure of a reasoning chain."""
    if not args:
        print("Usage: director-ai verify-reasoning <text>")
        sys.exit(1)

    from director_ai.core.verification.reasoning_verifier import verify_reasoning_chain

    result = verify_reasoning_chain(" ".join(args))
    print(f"Chain valid: {result.chain_valid}")
    print(f"Steps:       {result.steps_found}")
    print(f"Issues:      {result.issues_found}")
    for v in result.verdicts:
        print(f"  Step {v.step_index}: {v.verdict} ({v.confidence:.2f}) {v.reason}")


def _cmd_temporal_freshness(args: list[str]) -> None:
    """Score temporal freshness of claims."""
    if not args:
        print("Usage: director-ai temporal-freshness <text>")
        sys.exit(1)

    from director_ai.core.scoring.temporal_freshness import score_temporal_freshness

    result = score_temporal_freshness(" ".join(args))
    print(f"Has temporal claims: {result.has_temporal_claims}")
    print(f"Staleness risk:      {result.overall_staleness_risk:.2f}")
    print(f"Stale claims:        {len(result.stale_claims)}")
    for c in result.claims:
        print(f"  [{c.claim_type}] {c.text} (risk: {c.staleness_risk:.2f})")


def _cmd_check_step(args: list[str]) -> None:
    """Check an agentic step for safety issues."""
    if len(args) < 2:
        print("Usage: director-ai check-step <goal> <action> [args]")
        sys.exit(1)

    from director_ai.agentic.loop_monitor import LoopMonitor

    goal = args[0]
    action = args[1]
    action_args = args[2] if len(args) > 2 else ""

    monitor = LoopMonitor(goal=goal)
    verdict = monitor.check_step(action=action, args=action_args)
    print(f"Step:    {verdict.step_number}")
    print(f"Halt:    {verdict.should_halt}")
    print(f"Warn:    {verdict.should_warn}")
    print(f"Drift:   {verdict.goal_drift_score:.2f}")
    print(f"Budget:  {verdict.budget_remaining_pct:.0%}")
    if verdict.reasons:
        for r in verdict.reasons:
            print(f"  -> {r}")


def _cmd_consensus(args: list[str]) -> None:
    """Score factual agreement across multiple model responses."""
    if len(args) < 2:
        print(
            "Usage: director-ai consensus <model1:response1> <model2:response2> ...\n"
            "\n"
            "Each argument is model_name:response_text (colon-separated).\n"
            "Example: director-ai consensus 'gpt:Paris is the capital' 'claude:Paris is the capital'"
        )
        sys.exit(1)

    from director_ai.core.scoring.consensus import ConsensusScorer, ModelResponse

    responses = []
    for arg in args:
        if ":" not in arg:
            print(f"Invalid format: {arg!r} — expected model:response")
            sys.exit(1)
        model, _, response = arg.partition(":")
        responses.append(ModelResponse(model=model.strip(), response=response.strip()))

    scorer = ConsensusScorer(models=[r.model for r in responses])
    result = scorer.score_responses(responses)
    print(f"Models:    {result.num_models}")
    print(f"Agreement: {result.agreement_score:.2f}")
    print(f"Consensus: {result.has_consensus}")
    print(f"Lowest:    {result.lowest_pair_agreement:.2f}")
    for p in result.pairs:
        status = "agree" if p.agreed else "DISAGREE"
        print(f"  {p.model_a} vs {p.model_b}: {status} (divergence={p.divergence:.2f})")


def _cmd_kb_health(args: list[str]) -> None:
    """Run knowledge base health diagnostics."""
    from director_ai.core.config import DirectorConfig
    from director_ai.core.retrieval.kb_health import KBHealthCheck

    cfg = DirectorConfig.from_env()
    store = cfg.build_store()

    min_docs = 1
    max_latency = 100.0
    i = 0
    while i < len(args):
        if args[i] == "--min-docs" and i + 1 < len(args):
            min_docs = int(args[i + 1])
            i += 2
        elif args[i] == "--max-latency" and i + 1 < len(args):
            max_latency = float(args[i + 1])
            i += 2
        else:
            i += 1

    check = KBHealthCheck(
        store,
        min_documents=min_docs,
        max_query_latency_ms=max_latency,
    )
    report = check.run()

    print(report.summary)
    if report.issues:
        for issue in report.issues:
            print(f"  ISSUE: {issue}")
    if report.warnings:
        for warn in report.warnings:
            print(f"  WARNING: {warn}")

    sys.exit(0 if report.healthy else 1)


def _cmd_wizard(args: list[str]) -> None:
    """Launch the interactive configuration wizard."""
    cli_mode = "--cli" in args
    port = 7860
    share = "--share" in args
    output_path = None

    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        else:
            i += 1

    from director_ai.ui.config_wizard import launch_cli, launch_gradio

    if cli_mode:
        yaml_str = launch_cli()
        if output_path:
            from pathlib import Path

            Path(output_path).write_text(yaml_str, encoding="utf-8")
            print(f"\nConfig written to {output_path}")
    else:
        try:
            launch_gradio(port=port, share=share)
        except ImportError:
            print("Gradio not installed. Using CLI mode instead.")
            print("Install with: pip install director-ai[ui]\n")
            yaml_str = launch_cli()
            if output_path:
                from pathlib import Path

                Path(output_path).write_text(yaml_str, encoding="utf-8")
                print(f"\nConfig written to {output_path}")


def _cmd_cost_report(args: list[str]) -> None:
    """Show token cost report from the running scorer's CostAnalyser."""
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


def _cmd_adversarial_test(args: list[str]) -> None:
    """Run adversarial robustness test against the guardrail."""
    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    scorer = cfg.build_scorer()

    from director_ai.testing.adversarial_suite import AdversarialTester

    prompt = args[0] if args else "Tell me about this topic."

    def review_fn(p: str, r: str):
        approved, score = scorer.review(p, r)
        return approved, score.score

    tester = AdversarialTester(review_fn=review_fn, prompt=prompt)
    report = tester.run()
    print(f"Patterns:   {report.total_patterns}")
    print(f"Detected:   {report.detected}")
    print(f"Bypassed:   {report.bypassed}")
    print(f"Rate:       {report.detection_rate:.0%}")
    print(f"Robust:     {report.is_robust}")
    if report.vulnerable_categories:
        print(f"Vulnerable: {', '.join(report.vulnerable_categories)}")
