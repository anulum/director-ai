# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Command Line Interface

"""CLI entry point for Director-Class AI.

Usage::

    director-ai version
    director-ai review "What color is the sky?" "The sky is blue."
    director-ai process "What color is the sky?"
    director-ai batch input.jsonl --output results.jsonl
    director-ai bench --dataset regression --seed 42 --output results.json
    director-ai serve --port 8080 --profile thorough
    director-ai config --profile fast
"""

from __future__ import annotations

import json

# Used only for explicit --run, executing a fixed docker compose argv.
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# CLI subcommands extracted to reduce module size
from ._cli_bench import (
    _cmd_bench,
    _cmd_eval,
    _cmd_export,
    _cmd_finetune,
    _cmd_tune,
    _cmd_validate_data,
)
from ._cli_gate import _cmd_ci_gate
from ._cli_ingest import _INGEST_MAX_FILE_SIZE, _cmd_ingest
from ._cli_production import _cmd_production_check
from ._cli_train import _cmd_train

# Historical re-export — downstream tests and tooling reach for
# ``director_ai.cli._INGEST_MAX_FILE_SIZE``. Listing it in
# ``__all__`` documents the public surface so ruff does not flag
# the import as unused.
__all__ = ["_INGEST_MAX_FILE_SIZE"]
from ._cli_serve import _cmd_proxy, _cmd_serve, _cmd_stress_test
from ._cli_verify import (
    _cmd_adversarial_test,
    _cmd_check_step,
    _cmd_compliance,
    _cmd_consensus,
    _cmd_cost_report,
    _cmd_doctor,
    _cmd_forensics,
    _cmd_kb_health,
    _cmd_kpis,
    _cmd_license,
    _cmd_safety_dashboard,
    _cmd_temporal_freshness,
    _cmd_verify_numeric,
    _cmd_verify_reasoning,
    _cmd_wizard,
)
from .cli_quickstart import _cmd_quickstart

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class _CommandSpec:
    """Top-level CLI command metadata shared by dispatch and help output."""

    handler: Callable[[list[str]], None]
    help_line: str


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — dispatches to subcommands."""
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return

    import re

    cmd = args[0]
    if not re.match(r"^[a-z][a-z0-9-]*$", cmd):
        print("Invalid command name.")
        sys.exit(1)
    rest = args[1:]

    commands = _command_specs()

    if cmd not in commands:
        print("Unknown command.")
        _print_help()
        sys.exit(1)

    commands[cmd].handler(rest)


def _print_help() -> None:
    command_lines = "\n".join(
        f"  {spec.help_line}" for spec in _command_specs().values()
    )
    print(
        "Director-Class AI CLI\n"
        "\n"
        "Usage: director-ai <command> [options]\n"
        "\n"
        "Commands:\n"
        f"{command_lines}\n",
    )


def _cmd_version(args: list[str]) -> None:
    import platform

    import director_ai

    version_line = f"director-ai {director_ai.__version__}"
    print(version_line)
    print(
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}"
    )


def _cmd_evidence(args: list[str]) -> None:
    """Run the narrow grounding demo and emit a verifiable evidence packet."""
    out_dir = Path("evidence")
    profile = "fast"
    i = 0
    while i < len(args):
        if args[i] == "--emit" and i + 1 < len(args):
            out_dir = Path(args[i + 1])
            i += 2
        elif args[i] == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
            i += 2
        else:
            i += 1

    from director_ai.core.evidence_packet import build_evidence_packet
    from director_ai.guard import ProductionGuard

    guard = ProductionGuard.from_profile(profile)
    packet = build_evidence_packet(guard)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evidence_packet.json"
    path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")

    checks = packet["content"]["checks"]
    print(f"Evidence packet written to {path}")
    print(f"  grounded answer approved:    {checks['grounded_approved']}")
    print(f"  hallucinated answer blocked: {checks['hallucinated_blocked']}")
    print(f"  integrity (sha256): {packet['integrity']['digest'][:16]}…")
    if not (checks["grounded_approved"] and checks["hallucinated_blocked"]):
        print("Demo expectations not met (see packet).")
        sys.exit(1)


def _cmd_verify_evidence(args: list[str]) -> None:
    """Verify an evidence packet's integrity and demo expectations."""
    if not args:
        print("Usage: director-ai verify-evidence <packet.json|dir>")
        sys.exit(1)
    target = Path(args[0])
    if target.is_dir():
        target = target / "evidence_packet.json"
    if not target.exists():
        print(f"Error: evidence packet not found at {target}")
        sys.exit(1)

    from director_ai.core.evidence_packet import verify_evidence_packet

    packet = json.loads(target.read_text(encoding="utf-8"))
    ok, reason = verify_evidence_packet(packet)
    if ok:
        print(f"Evidence packet VERIFIED: {target}")
    else:
        print(f"Evidence packet INVALID ({reason}): {target}")
        sys.exit(1)


def _cmd_verify_audit(args: list[str]) -> None:
    """Verify the tamper-evident hash chain of an audit JSONL log.

    Usage: director-ai verify-audit <audit.jsonl> [--secret KEY]
    (the secret defaults to DIRECTOR_AUDIT_HMAC_SECRET; it must match the one
    that wrote the log for the HMAC tags to validate).
    """
    if not args:
        print("Usage: director-ai verify-audit <audit.jsonl> [--secret KEY]")
        sys.exit(1)
    path = Path(args[0])
    if not path.exists():
        print(f"Error: audit log not found at {path}")
        sys.exit(1)

    secret = None
    if "--secret" in args:
        idx = args.index("--secret")
        if idx + 1 < len(args):
            secret = args[idx + 1]

    from director_ai.core.safety.audit import AuditLogger

    ok, bad_index = AuditLogger(hmac_secret=secret).verify_chain(path)
    if ok:
        print(f"Audit chain VERIFIED: {path}")
    else:
        print(f"Audit chain TAMPERED at entry {bad_index}: {path}")
        sys.exit(1)


def _cmd_review(args: list[str]) -> None:
    if _is_help_request(args):
        _print_review_help()
        return

    if len(args) < 2:
        _print_review_help()
        sys.exit(1)

    prompt, response = args[0], args[1]

    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    scorer = cfg.build_scorer()
    approved, score = scorer.review(prompt, response)

    print(f"Approved:  {approved}")
    print(f"Coherence: {score.score:.4f}")
    print(f"H_logical: {score.h_logical:.4f}")
    print(f"H_factual: {score.h_factual:.4f}")


def _cmd_process(args: list[str]) -> None:
    if _is_help_request(args):
        _print_process_help()
        return

    if not args:
        _print_process_help()
        sys.exit(1)

    prompt = args[0]

    from director_ai.core.agent import CoherenceAgent
    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    store = cfg.build_store()
    scorer = cfg.build_scorer(store=store)
    agent = CoherenceAgent(
        _scorer=scorer, _store=store, max_candidates=cfg.max_candidates
    )
    result = agent.process(prompt)

    print(f"Output:     {result.output}")
    print(f"Halted:     {result.halted}")
    print(f"Candidates: {result.candidates_evaluated}")
    if result.coherence:  # pragma: no branch
        print(f"Coherence:  {result.coherence.score:.4f}")


_BATCH_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
_BATCH_MAX_LINE_SIZE = 1 * 1024 * 1024  # 1 MB per line
_BATCH_MAX_PROMPTS = 10_000


def _cmd_batch(args: list[str]) -> None:
    if _is_help_request(args):
        _print_batch_help()
        return

    if not args:
        _print_batch_help()
        sys.exit(1)

    input_file = args[0]
    output_file = None
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):  # pragma: no branch
            output_file = args[idx + 1]

    import os

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    file_size = os.path.getsize(input_file)
    if file_size > _BATCH_MAX_FILE_SIZE:
        print(
            f"Error: file too large ({file_size / 1024 / 1024:.1f} MB, "
            f"limit {_BATCH_MAX_FILE_SIZE // 1024 // 1024} MB)",
        )
        sys.exit(1)

    from director_ai.core.agent import CoherenceAgent
    from director_ai.core.runtime.batch import BatchProcessor

    prompts: list[str] = []
    with open(input_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if len(line) > _BATCH_MAX_LINE_SIZE:
                print(
                    f"Warning: skipping line {line_no} "
                    f"({len(line)} chars > {_BATCH_MAX_LINE_SIZE} limit)",
                )
                continue
            if len(prompts) >= _BATCH_MAX_PROMPTS:
                print(f"Warning: truncated at {_BATCH_MAX_PROMPTS} prompts")
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON on line {line_no}: {e}")
                continue
            prompt = data.get("prompt", data.get("text", line))
            if not isinstance(prompt, str) or not prompt.strip():
                print(f"Warning: skipping invalid prompt on line {line_no}")
                continue
            prompts.append(prompt)

    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    store = cfg.build_store()
    scorer = cfg.build_scorer(store=store)
    agent = CoherenceAgent(
        _scorer=scorer, _store=store, max_candidates=cfg.max_candidates
    )
    processor = BatchProcessor(agent)
    result = processor.process_batch(prompts)

    print(f"Total:    {result.total}")
    print(f"Success:  {result.succeeded}")
    print(f"Failed:   {result.failed}")
    print(f"Duration: {result.duration_seconds:.2f}s")

    if output_file:
        from director_ai.core.types import ReviewResult

        with open(output_file, "w", encoding="utf-8") as f:
            for r in result.results:
                if not isinstance(r, ReviewResult):
                    # BatchResult.results can also hold
                    # (approved, score) tuples from the review
                    # path; skip those — the CLI --out flag
                    # only serialises full ReviewResult records.
                    continue
                f.write(
                    json.dumps(
                        {
                            "output": r.output,
                            "halted": r.halted,
                            "coherence": (r.coherence.score if r.coherence else None),
                        },
                    )
                    + "\n",
                )
        print(f"Results written to {output_file}")


def _cmd_config(args: list[str]) -> None:
    if _is_help_request(args):
        _print_config_help()
        return

    from director_ai.core.config import DirectorConfig

    if "--profile" in args:
        idx = args.index("--profile")
        if idx + 1 < len(args):
            cfg = DirectorConfig.from_profile(args[idx + 1])
        else:
            _print_config_help()
            sys.exit(1)
    else:
        cfg = DirectorConfig.from_env()

    for key, value in cfg.to_dict().items():
        print(f"  {key}: {value}")


def _command_specs() -> dict[str, _CommandSpec]:
    """Return the ordered top-level CLI registry used for dispatch and help."""
    return {
        "version": _CommandSpec(
            _cmd_version, "version               Show version info"
        ),
        "quickstart": _CommandSpec(
            _cmd_quickstart,
            "quickstart [--profile P] [--run]  Scaffold a working project",
        ),
        "production-check": _CommandSpec(
            _cmd_production_check,
            "production-check [--path D]  Validate a generated production scaffold",
        ),
        "review": _CommandSpec(
            _cmd_review,
            "review <prompt> <resp> Review a prompt/response pair",
        ),
        "process": _CommandSpec(
            _cmd_process,
            "process <prompt>      Process a prompt through the full pipeline",
        ),
        "batch": _CommandSpec(
            _cmd_batch,
            "batch <file.jsonl>    Batch process (max 10K prompts, <100MB)",
        ),
        "ingest": _CommandSpec(
            _cmd_ingest,
            "ingest <file-or-dir> [--persist DIR] [--chunk-size N]  "
            "Ingest txt/md/pdf/docx/html/csv",
        ),
        "eval": _CommandSpec(
            _cmd_eval, "eval [--dataset D]    Run NLI benchmark suite"
        ),
        "ci-gate": _CommandSpec(
            _cmd_ci_gate,
            "ci-gate --dataset F --min-accuracy R  Fail CI when guard quality drops",
        ),
        "bench": _CommandSpec(
            _cmd_bench,
            "bench [--dataset D] [--seed N] [--output F]  Run regression benchmarks",
        ),
        "tune": _CommandSpec(
            _cmd_tune,
            "tune <file.jsonl>|--dataset D [--output config.yaml]  "
            "Tune profile overlay",
        ),
        "train": _CommandSpec(
            _cmd_train,
            "train submit [options]  Submit or dry-run a managed training job",
        ),
        "finetune": _CommandSpec(
            _cmd_finetune,
            "finetune <train.jsonl> [options]  Fine-tune NLI model on domain data",
        ),
        "validate-data": _CommandSpec(
            _cmd_validate_data,
            "validate-data <file.jsonl>       Validate data before fine-tuning",
        ),
        "serve": _CommandSpec(
            _cmd_serve,
            "serve [--port N] [--transport http|grpc] [--workers W] "
            "[--dev|--production]  Start the server",
        ),
        "proxy": _CommandSpec(
            _cmd_proxy,
            "proxy [--port N] [--facts F]   Chat-completions guardrail proxy",
        ),
        "export": _CommandSpec(
            _cmd_export,
            "export [--format F]   Export model to ONNX/TensorRT",
        ),
        "stress-test": _CommandSpec(
            _cmd_stress_test,
            "stress-test [options] Benchmark streaming kernel throughput",
        ),
        "doctor": _CommandSpec(
            _cmd_doctor,
            "doctor                Check runtime dependencies and readiness",
        ),
        "license": _CommandSpec(
            _cmd_license,
            "license [status|generate|validate|polar-env]  Commercial license tools",
        ),
        "config": _CommandSpec(
            _cmd_config,
            "config [--profile X]  Show/set configuration",
        ),
        "compliance": _CommandSpec(
            _cmd_compliance,
            "compliance <sub>      EU AI Act compliance (report, status, drift)",
        ),
        "verify-numeric": _CommandSpec(
            _cmd_verify_numeric,
            "verify-numeric <text> Check numeric consistency in text",
        ),
        "verify-reasoning": _CommandSpec(
            _cmd_verify_reasoning,
            "verify-reasoning <text> Check logical reasoning structure",
        ),
        "temporal-freshness": _CommandSpec(
            _cmd_temporal_freshness,
            "temporal-freshness <text> Check date-sensitive claim freshness",
        ),
        "check-step": _CommandSpec(
            _cmd_check_step,
            "check-step <goal> <action> [args]  Validate an agentic loop step",
        ),
        "consensus": _CommandSpec(
            _cmd_consensus,
            "consensus <model:response>...  Compare cross-model agreement",
        ),
        "adversarial-test": _CommandSpec(
            _cmd_adversarial_test,
            "adversarial-test [prompt]  Run guardrail adversarial probes",
        ),
        "kb-health": _CommandSpec(
            _cmd_kb_health,
            "kb-health [options]   Knowledge base health diagnostics",
        ),
        "wizard": _CommandSpec(
            _cmd_wizard,
            "wizard [--cli]        Interactive configuration wizard",
        ),
        "safety-dashboard": _CommandSpec(
            _cmd_safety_dashboard,
            "safety-dashboard [--text] [--events F]  Halt-rate operations view",
        ),
        "kpis": _CommandSpec(
            _cmd_kpis,
            "kpis --input F [--format text|markdown|json]  Board-level guardrail KPIs",
        ),
        "forensics": _CommandSpec(
            _cmd_forensics,
            "forensics --input F [--format text|markdown|json]  Scorer-miss forensics",
        ),
        "cost-report": _CommandSpec(
            _cmd_cost_report,
            "cost-report [--format F]  Token cost report (text|json|html)",
        ),
        "evidence": _CommandSpec(
            _cmd_evidence,
            "evidence [--emit DIR]  Run the narrow demo, emit a verifiable packet",
        ),
        "verify-evidence": _CommandSpec(
            _cmd_verify_evidence,
            "verify-evidence <DIR>  Verify an emitted evidence packet",
        ),
        "verify-audit": _CommandSpec(
            _cmd_verify_audit,
            "verify-audit <FILE>  Verify an audit log's tamper-evident hash chain",
        ),
    }


def _is_help_request(args: list[str]) -> bool:
    """Return whether a command received an explicit help token."""
    return bool(args) and args[0] in ("-h", "--help", "help")


def _print_review_help() -> None:
    """Print review command usage without building the scorer."""
    print(
        "Usage: director-ai review <prompt> <response>\n"
        "\n"
        "Review one prompt/response pair with the configured scorer.\n"
    )


def _print_process_help() -> None:
    """Print process command usage without building the agent pipeline."""
    print(
        "Usage: director-ai process <prompt>\n"
        "\n"
        "Generate and review one response through the configured agent pipeline.\n"
    )


def _print_batch_help() -> None:
    """Print batch command usage without opening an input file."""
    print(
        "Usage: director-ai batch <input.jsonl> [--output results.jsonl]\n"
        "\n"
        "Batch process JSONL prompts. Each input line should include a prompt field.\n"
        "Limits: 10K prompts, 100 MB file, 1 MB per line.\n"
        "\n"
        "Options:\n"
        "  --output FILE          Write full ReviewResult records as JSONL\n"
    )


def _print_config_help() -> None:
    """Print config command usage without loading environment configuration."""
    print(
        "Usage: director-ai config [--profile <name>]\n"
        "\n"
        "Print the active configuration or a built-in profile configuration.\n"
        "\n"
        "Options:\n"
        "  --profile NAME         Print a built-in profile instead of env config\n"
    )


if __name__ == "__main__":
    main()
