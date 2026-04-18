# SPDX-License-Identifier: AGPL-3.0-or-later
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
import sys

# CLI subcommands extracted to reduce module size
from ._cli_bench import (
    _cmd_bench,
    _cmd_eval,
    _cmd_export,
    _cmd_finetune,
    _cmd_tune,
    _cmd_validate_data,
)
from ._cli_ingest import _INGEST_MAX_FILE_SIZE, _cmd_ingest

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
    _cmd_kb_health,
    _cmd_license,
    _cmd_temporal_freshness,
    _cmd_verify_numeric,
    _cmd_verify_reasoning,
    _cmd_wizard,
)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — dispatches to subcommands."""
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return

    import re

    cmd = args[0]
    if not re.match(r"^[a-z][a-z0-9-]*$", cmd):
        print(f"Invalid command name: {cmd!r}")
        sys.exit(1)
    rest = args[1:]

    commands = {
        "version": _cmd_version,
        "quickstart": _cmd_quickstart,
        "review": _cmd_review,
        "process": _cmd_process,
        "batch": _cmd_batch,
        "ingest": _cmd_ingest,
        "eval": _cmd_eval,
        "bench": _cmd_bench,
        "tune": _cmd_tune,
        "finetune": _cmd_finetune,
        "validate-data": _cmd_validate_data,
        "export": _cmd_export,
        "serve": _cmd_serve,
        "proxy": _cmd_proxy,
        "config": _cmd_config,
        "stress-test": _cmd_stress_test,
        "doctor": _cmd_doctor,
        "license": _cmd_license,
        "compliance": _cmd_compliance,
        "verify-numeric": _cmd_verify_numeric,
        "verify-reasoning": _cmd_verify_reasoning,
        "temporal-freshness": _cmd_temporal_freshness,
        "check-step": _cmd_check_step,
        "consensus": _cmd_consensus,
        "adversarial-test": _cmd_adversarial_test,
        "kb-health": _cmd_kb_health,
        "wizard": _cmd_wizard,
        "cost-report": _cmd_cost_report,
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(1)

    commands[cmd](rest)


def _print_help() -> None:
    print(
        "Director-Class AI CLI\n"
        "\n"
        "Usage: director-ai <command> [options]\n"
        "\n"
        "Commands:\n"
        "  version               Show version info\n"
        "  quickstart [--profile P]  Scaffold a working project\n"
        "  review <prompt> <resp> Review a prompt/response pair\n"
        "  process <prompt>      Process a prompt through the full pipeline\n"
        "  batch <file.jsonl>    Batch process (max 10K prompts, <100MB)\n"
        "  ingest <file>         Ingest documents (txt/md/pdf/docx/html/csv)\n"
        "  eval [--dataset D]    Run NLI benchmark suite\n"
        "  bench [--dataset D] [--seed N] [--output F]  Run regression benchmarks\n"
        "  tune <file.jsonl> [--output config.yaml]  Find optimal threshold\n"
        "  finetune <train.jsonl> [options]  Fine-tune NLI model on domain data\n"
        "  validate-data <file.jsonl>       Validate data before fine-tuning\n"
        "  serve [--port N] [--workers W]  Start the FastAPI server\n"
        "  proxy [--port N] [--facts F]   OpenAI-compatible guardrail proxy\n"
        "  export [--format F]   Export model to ONNX/TensorRT\n"
        "  stress-test [options] Benchmark streaming kernel throughput\n"
        "  doctor                Check runtime dependencies and readiness\n"
        "  config [--profile X]  Show/set configuration\n"
        "  compliance <sub>      EU AI Act compliance (report, status, drift)\n"
        "  kb-health [options]   Knowledge base health diagnostics\n"
        "  wizard [--cli]        Interactive configuration wizard\n"
        "  cost-report [--format F]  Token cost report (text|json|html)\n",
    )


def _cmd_version(args: list[str]) -> None:
    import platform

    import director_ai

    version_line = f"director-ai {director_ai.__version__}"
    print(version_line)
    print(
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}"
    )


_VALID_PROFILES = (
    "medical",
    "finance",
    "legal",
    "creative",
    "customer_support",
    "summarization",
    "fast",
    "thorough",
    "research",
    "lite",
)


def _cmd_quickstart(args: list[str]) -> None:
    """Scaffold a working director-ai project in one command."""
    from pathlib import Path

    from director_ai.core.config import DirectorConfig

    profile = "fast"
    i = 0
    while i < len(args):
        if args[i] == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
            i += 2
        else:
            i += 1

    if profile not in _VALID_PROFILES:
        print(f"Unknown profile '{profile}'. Choose from: {', '.join(_VALID_PROFILES)}")
        sys.exit(1)

    out_dir = Path("director_guard")
    if out_dir.exists():
        print(f"Error: {out_dir}/ already exists. Remove it or use a new dir.")
        sys.exit(1)

    cfg = DirectorConfig.from_profile(profile)
    out_dir.mkdir()

    # config.yaml
    (out_dir / "config.yaml").write_text(
        f"# Director-AI configuration — profile: {profile}\n"
        f"coherence_threshold: {cfg.coherence_threshold}\n"
        f"hard_limit: {cfg.hard_limit}\n"
        f"use_nli: {str(cfg.use_nli).lower()}\n"
        f"profile: {profile}\n",
        encoding="utf-8",
    )

    # facts.txt
    (out_dir / "facts.txt").write_text(
        "The sky is blue due to Rayleigh scattering.\n"
        "Water boils at 100 degrees Celsius at sea level.\n"
        "The Earth orbits the Sun once every 365.25 days.\n",
        encoding="utf-8",
    )

    # guard.py
    (out_dir / "guard.py").write_text(
        '"""Minimal Director-AI guard — run: python guard.py"""\n'
        "from pathlib import Path\n"
        "\n"
        "from director_ai.core import CoherenceScorer, GroundTruthStore\n"
        "from director_ai.core.config import DirectorConfig\n"
        "\n"
        "_HERE = Path(__file__).resolve().parent\n"
        "config = DirectorConfig.from_yaml(str(_HERE / 'config.yaml'))\n"
        "store = GroundTruthStore()\n"
        "with open(_HERE / 'facts.txt') as f:\n"
        "    for line in f:\n"
        "        line = line.strip()\n"
        "        if line:\n"
        "            store.add(line[:20], line)\n"
        "\n"
        "scorer = CoherenceScorer(\n"
        "    threshold=config.coherence_threshold,\n"
        "    ground_truth_store=store,\n"
        "    use_nli=config.use_nli,\n"
        ")\n"
        "\n"
        "approved, score = scorer.review(\n"
        '    "What color is the sky?", "The sky is blue."\n'
        ")\n"
        'print(f"Approved: {approved}  Score: {score.score:.3f}")\n',
        encoding="utf-8",
    )

    # README.md
    (out_dir / "README.md").write_text(
        f"# Director-AI Guard (profile: {profile})\n"
        "\n"
        "```bash\n"
        "pip install director-ai\n"
        "python guard.py\n"
        "```\n"
        "\n"
        "Edit `facts.txt` to add your own knowledge base.\n"
        "Edit `config.yaml` to tune thresholds.\n",
        encoding="utf-8",
    )

    print(f"Created {out_dir}/ — run: python {out_dir}/guard.py")


def _cmd_review(args: list[str]) -> None:
    if len(args) < 2:
        print("Usage: director-ai review <prompt> <response>")
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
    if not args:
        print("Usage: director-ai process <prompt>")
        sys.exit(1)

    prompt = args[0]

    from director_ai.core.agent import CoherenceAgent
    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    store = cfg.build_store()
    scorer = cfg.build_scorer(store=store)
    agent = CoherenceAgent(_scorer=scorer, _store=store)
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
    if not args:
        print("Usage: director-ai batch <input.jsonl> [--output results.jsonl]")
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
    agent = CoherenceAgent(_scorer=scorer, _store=store)
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
    from director_ai.core.config import DirectorConfig

    if "--profile" in args:
        idx = args.index("--profile")
        if idx + 1 < len(args):
            cfg = DirectorConfig.from_profile(args[idx + 1])
        else:
            print("Usage: director-ai config --profile <name>")
            sys.exit(1)
    else:
        cfg = DirectorConfig.from_env()

    for key, value in cfg.to_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
