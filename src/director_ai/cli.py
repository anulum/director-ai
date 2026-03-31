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
        "  compliance <sub>      EU AI Act compliance (report, status, drift)\n",
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
        with open(output_file, "w", encoding="utf-8") as f:
            for r in result.results:
                f.write(
                    json.dumps(
                        {
                            "output": r.output,  # type: ignore[union-attr]
                            "halted": r.halted,  # type: ignore[union-attr]
                            "coherence": r.coherence.score if r.coherence else None,  # type: ignore[union-attr]
                        },
                    )
                    + "\n",
                )
        print(f"Results written to {output_file}")


_INGEST_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def _cmd_ingest(args: list[str]) -> None:
    """Ingest files or directories into a VectorGroundTruthStore.

    Supported formats: ``.txt``, ``.md``, ``.json``/``.jsonl``,
    ``.pdf``, ``.docx``, ``.html``, ``.csv``.
    PDF/DOCX/HTML require ``pip install director-ai[ingestion]``.
    Directories are walked recursively for supported file types.
    """
    if not args:
        print(
            "Usage: director-ai ingest <file-or-dir> "
            "[--persist <dir>] [--chunk-size <tokens>]",
        )
        sys.exit(1)

    import os
    from pathlib import Path

    input_path = args[0]
    persist_dir = None
    chunk_size = 500
    if "--persist" in args:
        idx = args.index("--persist")
        if idx + 1 < len(args):  # pragma: no branch
            persist_dir = args[idx + 1]
    if "--chunk-size" in args:
        idx = args.index("--chunk-size")
        if idx + 1 < len(args):  # pragma: no branch
            chunk_size = int(args[idx + 1])
    if chunk_size <= 0:
        print(f"Error: --chunk-size must be > 0, got {chunk_size}")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: path not found: {input_path}")
        sys.exit(1)

    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    if persist_dir:
        cfg.vector_backend = "chroma"
        cfg.chroma_persist_dir = persist_dir
    store = cfg.build_store()

    text_exts = {".txt", ".md", ".json", ".jsonl", ".xml", ".markdown"}
    parsed_exts = {".pdf", ".docx", ".html", ".htm", ".csv"}
    supported_exts = text_exts | parsed_exts

    def _collect_files(path: str) -> list[Path]:
        p = Path(path)
        if p.is_file():
            return [p]
        return sorted(
            f
            for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_exts
        )

    def _chunk_paragraphs(text: str, max_tokens: int) -> list[str]:
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            para = para.strip()
            if not para:  # pragma: no cover — empty paragraphs from text.split
                continue
            word_count = len(para.split())
            if current_len + word_count > max_tokens and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += word_count
        if current:  # pragma: no branch
            chunks.append("\n\n".join(current))
        return chunks

    def _read_file(path: Path) -> list[str]:
        size = path.stat().st_size
        if size > _INGEST_MAX_FILE_SIZE:
            print(f"Warning: skipping {path} ({size / 1024 / 1024:.1f} MB > limit)")
            return []

        ext = path.suffix.lower()

        # PDF, DOCX, HTML, CSV — delegate to doc_parser (binary read)
        if ext in parsed_exts:
            from director_ai.core.retrieval.doc_parser import parse

            try:
                raw = path.read_bytes()
                text = parse(raw, path.name)
            except ImportError as exc:
                print(f"Warning: skipping {path} ({exc})")
                return []
            if not text.strip():
                return []
            return _chunk_paragraphs(text, chunk_size)

        text = path.read_text(encoding="utf-8", errors="replace")

        if ext in (".json", ".jsonl"):
            docs = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc = data.get("text", data.get("content", ""))
                    if doc:  # pragma: no branch
                        docs.append(doc)
                except json.JSONDecodeError:
                    pass
            return docs
        return _chunk_paragraphs(text, chunk_size)

    files = _collect_files(input_path)
    if not files:
        print(f"No supported files found in {input_path}")
        sys.exit(1)

    texts: list[str] = []
    for f in files:
        texts.extend(_read_file(f))

    count = store.ingest(texts)
    print(f"Ingested {count} chunks from {len(files)} file(s).")
    if persist_dir:
        print(f"Persisted to: {persist_dir}")
    else:
        print("(in-memory only — use --persist <dir> to save)")


def _cmd_eval(args: list[str]) -> None:
    """Run NLI benchmark suite.

    Usage::

        director-ai eval --dataset aggrefact --max-samples 100 --output results.json
    """
    import logging
    import os

    dataset = None
    max_samples = None
    output_file = None
    model = None
    quantize_mode = None

    i = 0
    while i < len(args):
        if args[i] == "--dataset" and i + 1 < len(args):
            dataset = args[i + 1]
            i += 2
        elif args[i] == "--max-samples" and i + 1 < len(args):
            try:
                max_samples = int(args[i + 1])
            except ValueError:
                print(f"Error: invalid --max-samples value: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--quantize" and i + 1 < len(args):
            quantize_mode = args[i + 1]
            if quantize_mode not in ("int8", "fp16"):
                print(
                    f"Error: --quantize must be 'int8' or 'fp16', got '{quantize_mode}'",
                )
                sys.exit(1)
            i += 2
        else:
            i += 1

    if quantize_mode:
        from director_ai.core.scoring.nli import export_onnx

        print(f"Exporting ONNX with {quantize_mode} quantization...")
        export_onnx(quantize=quantize_mode)
        print("Export complete.")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from benchmarks.run_all import _print_comparison_table, _run_suite
    except ImportError:
        print(
            "Error: benchmarks package not found. "
            "Run from the director-ai repo root, or install in editable mode.",
        )
        sys.exit(1)

    if dataset and dataset == "aggrefact" and not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not set — AggreFact benchmark may be skipped")

    print(
        f"Running benchmarks (max_samples={max_samples}, model={model or 'default'})...",
    )
    results = _run_suite(model, max_samples)
    _print_comparison_table({model or "default": results})

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {output_file}")


def _cmd_bench(args: list[str]) -> None:
    """Run regression benchmark suite.

    Usage::

        director-ai bench
        director-ai bench --dataset regression --seed 42 --output results.json
    """
    import random

    dataset = "regression"
    seed = None
    output_file = None
    max_samples = None

    i = 0
    while i < len(args):
        if args[i] == "--dataset" and i + 1 < len(args):
            dataset = args[i + 1]
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            try:
                seed = int(args[i + 1])
            except ValueError:
                print(f"Error: invalid --seed value: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--max-samples" and i + 1 < len(args):
            try:
                max_samples = int(args[i + 1])
            except ValueError:
                print(f"Error: invalid --max-samples value: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        else:
            i += 1

    if seed is not None:
        random.seed(seed)

    valid_datasets = ("regression", "e2e", "streaming")
    if dataset not in valid_datasets:
        print(f"Unknown dataset '{dataset}'. Choose from: {', '.join(valid_datasets)}")
        sys.exit(1)

    try:
        from benchmarks import regression_suite
    except ImportError:
        print(
            "Error: benchmarks package not found. "
            "Run from the director-ai repo root, or install in editable mode.",
        )
        sys.exit(1)

    print(f"Running benchmarks (dataset={dataset}, seed={seed})...")

    import time

    t0 = time.perf_counter()
    tests = {
        "regression": [
            regression_suite.test_heuristic_accuracy,
            regression_suite.test_streaming_stability,
            regression_suite.test_latency_ceiling,
            regression_suite.test_metrics_integrity,
            regression_suite.test_evidence_schema,
        ],
        "e2e": [
            regression_suite.test_e2e_heuristic_delta,
        ],
        "streaming": [
            regression_suite.test_false_halt_rate,
            regression_suite.test_streaming_stability,
        ],
    }

    suite = tests[dataset]
    if max_samples and len(suite) > max_samples:
        suite = suite[:max_samples]

    warn_only = {
        "regression": {"test_latency_ceiling"},
        "e2e": set(),
        "streaming": set(),
    }
    passed = 0
    failed = 0
    warned = 0
    results = []
    for test_fn in suite:
        try:
            test_fn()
            passed += 1
            results.append({"test": test_fn.__name__, "status": "passed"})
        except AssertionError as e:
            if test_fn.__name__ in warn_only.get(dataset, set()):
                warned += 1
                results.append(
                    {"test": test_fn.__name__, "status": "warned", "error": str(e)},
                )
                print(f"  WARN: {test_fn.__name__}: {e}")
                continue
            failed += 1
            results.append(
                {"test": test_fn.__name__, "status": "failed", "error": str(e)},
            )
            print(f"  FAIL: {test_fn.__name__}: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\n  {passed} passed, {warned} warned, {failed} failed in {elapsed:.2f}s")

    report = {
        "dataset": dataset,
        "seed": seed,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "duration_s": round(elapsed, 3),
        "results": results,
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Results written to {output_file}")

    if failed > 0:
        sys.exit(1)


def _cmd_tune(args: list[str]) -> None:
    """Find optimal threshold via grid search over labeled JSONL data.

    Input format: one JSON object per line with
    ``{"prompt": str, "response": str, "label": bool}``.
    """
    if not args:
        print("Usage: director-ai tune <labeled.jsonl> [--output config.yaml]")
        sys.exit(1)

    import os

    input_file = args[0]
    output_file = None
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):  # pragma: no branch
            output_file = args[idx + 1]

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    samples: list[dict] = []
    with open(input_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {line_no}: {e}")
                continue
            if "prompt" not in data or "response" not in data or "label" not in data:
                print(f"Warning: skipping line {line_no}: missing required fields")
                continue
            samples.append(data)

    if not samples:
        print("Error: no valid samples found")
        sys.exit(1)

    from director_ai.core.training.tuner import tune

    result = tune(samples)

    print(f"Best threshold: {result.threshold}")
    print(f"Weights:        w_logic={result.w_logic}, w_fact={result.w_fact}")
    print(f"Balanced Acc:   {result.balanced_accuracy:.4f}")
    print(f"Precision:      {result.precision:.4f}")
    print(f"Recall:         {result.recall:.4f}")
    print(f"F1:             {result.f1:.4f}")
    print(f"Samples:        {result.samples}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(
                f"coherence_threshold: {result.threshold}\n"
                f"w_logic: {result.w_logic}\n"
                f"w_fact: {result.w_fact}\n",
            )
        print(f"Config written to {output_file}")


def _cmd_finetune(args: list[str]) -> None:
    """Fine-tune NLI model on domain-specific labeled data.

    Input JSONL format: ``{"premise": str, "hypothesis": str, "label": int}``
    Also accepts: ``{"doc": str, "claim": str, "label": int}``
    """
    if not args:
        print(
            "Usage: director-ai finetune <train.jsonl> [options]\n"
            "\n"
            "Options:\n"
            "  --eval <eval.jsonl>  Evaluation data\n"
            "  --output <dir>      Output directory (default: ./director-finetuned)\n"
            "  --epochs N          Training epochs (default: 3)\n"
            "  --lr FLOAT          Learning rate (default: 2e-5)\n"
            "  --batch-size N      Batch size (default: 16)\n"
            "  --base-model ID     Base model (default: FactCG-DeBERTa-v3-Large)\n"
            "  --mix-general       Mix 20% general NLI data to prevent forgetting\n"
            "  --general-data P    Path to general NLI JSONL (for --mix-general)\n"
            "  --early-stopping N  Stop after N evals without improvement\n"
            "  --class-weights     Apply inverse-frequency class weights\n"
            "  --auto-benchmark    Run anti-regression check after training\n"
            "  --auto-onnx         Export to ONNX after training\n",
        )
        sys.exit(1)

    import os

    train_file = args[0]
    if not os.path.isfile(train_file):
        print(f"Error: file not found: {train_file}")
        sys.exit(1)

    from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

    config = FinetuneConfig()
    eval_file = None

    i = 1
    while i < len(args):
        if args[i] == "--eval" and i + 1 < len(args):
            eval_file = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            config.output_dir = args[i + 1]
            i += 2
        elif args[i] == "--epochs" and i + 1 < len(args):
            config.epochs = int(args[i + 1])
            i += 2
        elif args[i] == "--lr" and i + 1 < len(args):
            config.learning_rate = float(args[i + 1])
            i += 2
        elif args[i] == "--batch-size" and i + 1 < len(args):
            config.batch_size = int(args[i + 1])
            i += 2
        elif args[i] == "--base-model" and i + 1 < len(args):
            config.base_model = args[i + 1]
            i += 2
        elif args[i] == "--mix-general":
            config.mix_general_data = True
            i += 1
        elif args[i] == "--general-data" and i + 1 < len(args):
            config.general_data_path = args[i + 1]
            config.mix_general_data = True
            i += 2
        elif args[i] == "--early-stopping" and i + 1 < len(args):
            config.early_stopping_patience = int(args[i + 1])
            i += 2
        elif args[i] == "--class-weights":
            config.class_weighted_loss = True
            i += 1
        elif args[i] == "--auto-benchmark":
            config.auto_benchmark = True
            i += 1
        elif args[i] == "--auto-onnx":
            config.auto_onnx_export = True
            i += 1
        else:
            print(f"Unknown option: {args[i]}")
            sys.exit(1)

    result = finetune_nli(train_file, eval_path=eval_file, config=config)

    print("\nFine-tuning complete.")
    print(f"  Model saved to:  {result.output_dir}")
    print(f"  Train samples:   {result.train_samples}")
    if result.mixed_general_samples:
        print(f"  Mixed general:   {result.mixed_general_samples}")
    print(f"  Epochs:          {result.epochs_completed}")
    print(f"  Final loss:      {result.final_loss:.4f}")
    if result.eval_samples:
        print(f"  Eval samples:    {result.eval_samples}")
        print(f"  Best bal. acc:   {result.best_balanced_accuracy:.1%}")
    if result.regression_report:
        rr = result.regression_report
        print(
            f"  Regression:      {rr['regression_pp']:+.1f}pp â†’ {rr['recommendation']}",
        )
    if result.onnx_path:
        print(f"  ONNX export:     {result.onnx_path}")
    print("\nUse the model:")
    print(f'  scorer = NLIScorer(model_name="{result.output_dir}")')


def _cmd_validate_data(args: list[str]) -> None:
    """Validate JSONL data before fine-tuning."""
    if not args:
        print("Usage: director-ai validate-data <file.jsonl>")
        sys.exit(1)

    import os

    data_file = args[0]
    if not os.path.isfile(data_file):
        print(f"Error: file not found: {data_file}")
        sys.exit(1)

    from director_ai.core.training.finetune_validator import validate_finetune_data

    report = validate_finetune_data(data_file)
    print(report.summary())

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.errors:
        print("\nErrors:")
        for e in report.errors:
            print(f"  - {e}")
        sys.exit(1)


def _cmd_export(args: list[str]) -> None:
    """Export NLI model to ONNX or pre-build TensorRT engine cache.

    Usage::

        director-ai export --format onnx --output factcg_onnx
        director-ai export --format onnx --quantize int8
        director-ai export --format tensorrt --onnx-dir factcg_onnx
    """
    fmt = "onnx"
    output_dir = "factcg_onnx"
    onnx_dir = "factcg_onnx"
    quantize = None
    fp16 = True
    model = None

    i = 0
    while i < len(args):
        if args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        elif args[i] == "--onnx-dir" and i + 1 < len(args):
            onnx_dir = args[i + 1]
            i += 2
        elif args[i] == "--quantize" and i + 1 < len(args):
            quantize = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--no-fp16":
            fp16 = False
            i += 1
        else:
            i += 1

    if fmt == "onnx":
        from director_ai.core.scoring.nli import export_onnx

        print(f"Exporting ONNX model to {output_dir}...")
        export_onnx(
            model_name=model or "yaxili96/FactCG-DeBERTa-v3-Large",
            output_dir=output_dir,
            quantize=quantize,
        )
        print(f"Done. Load with: NLIScorer(backend='onnx', onnx_path='{output_dir}')")
    elif fmt == "tensorrt":
        from director_ai.core.scoring.nli import export_tensorrt

        print(f"Building TensorRT engine cache from {onnx_dir}...")
        cache_dir = export_tensorrt(onnx_dir=onnx_dir, output_dir=output_dir, fp16=fp16)
        print(f"Done. Cache at {cache_dir}. TRT will auto-activate on next load.")
    else:
        print(f"Unknown format '{fmt}'. Use 'onnx' or 'tensorrt'.")
        sys.exit(1)


def _cmd_proxy(args: list[str]) -> None:
    """Start OpenAI-compatible guardrail proxy."""
    port = 8080
    threshold = 0.6
    facts_path = None
    upstream_url = "https://api.openai.com"
    on_fail = "reject"
    api_keys: list[str] = []
    allow_http = False
    audit_db: str | None = None

    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--threshold" and i + 1 < len(args):
            threshold = float(args[i + 1])
            i += 2
        elif args[i] == "--facts" and i + 1 < len(args):
            facts_path = args[i + 1]
            i += 2
        elif args[i] == "--upstream-url" and i + 1 < len(args):
            upstream_url = args[i + 1]
            i += 2
        elif args[i] == "--on-fail" and i + 1 < len(args):
            on_fail = args[i + 1]
            if on_fail not in ("reject", "warn"):
                print(f"Error: --on-fail must be 'reject' or 'warn', got '{on_fail}'")
                sys.exit(1)
            i += 2
        elif args[i] == "--api-keys" and i + 1 < len(args):
            api_keys = [k.strip() for k in args[i + 1].split(",") if k.strip()]
            i += 2
        elif args[i] == "--allow-http-upstream":
            allow_http = True
            i += 1
        elif args[i] == "--audit-db" and i + 1 < len(args):
            audit_db = args[i + 1]
            i += 2
        else:
            i += 1

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install director-ai[server]")
        sys.exit(1)

    from director_ai.proxy import create_proxy_app

    app = create_proxy_app(
        threshold=threshold,
        facts_path=facts_path,
        upstream_url=upstream_url,
        on_fail=on_fail,
        api_keys=api_keys or None,
        allow_http_upstream=allow_http,
        audit_db=audit_db,
    )

    print(
        f"Director-AI proxy on :{port} â†’ {upstream_url} "
        f"(threshold={threshold}, on_fail={on_fail})",
    )
    uvicorn.run(app, host="0.0.0.0", port=port)


def _cmd_serve(args: list[str]) -> None:
    port = 8080
    host = "0.0.0.0"
    profile = "default"
    mode = ""
    workers = 1
    transport = "http"
    cors_origins = ""

    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            if mode not in ("general", "grounded", "auto"):
                print(
                    f"Error: --mode must be 'general', 'grounded', or 'auto', got '{mode}'"
                )
                sys.exit(1)
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"Error: invalid port number: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
            i += 2
        elif args[i] == "--workers" and i + 1 < len(args):
            try:
                workers = int(args[i + 1])
                if workers < 1:
                    raise ValueError
            except ValueError:
                print(f"Error: invalid worker count: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--cors-origins" and i + 1 < len(args):
            cors_origins = args[i + 1]
            i += 2
        elif args[i] == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
            if transport not in ("http", "grpc"):
                print(f"Error: --transport must be 'http' or 'grpc', got '{transport}'")
                sys.exit(1)
            i += 2
        else:
            i += 1

    from director_ai.core.config import DirectorConfig

    if profile != "default":
        config = DirectorConfig.from_profile(profile)
    else:
        config = DirectorConfig.from_env()
    if mode:
        config = DirectorConfig(**{**config.__dict__, "mode": mode})
    config.server_host = host
    config.server_port = port
    if cors_origins:
        config.cors_origins = cors_origins

    if transport == "grpc":
        from director_ai.grpc_server import create_grpc_server

        print(f"Starting Director AI gRPC server on port {port} (workers={workers})")
        server = create_grpc_server(config, max_workers=workers, port=port)
        server.start()
        server.wait_for_termination()
        return

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install director-ai[server]")
        sys.exit(1)

    from director_ai.server import create_app

    print(
        f"Starting Director AI server on {host}:{port} "
        f"(profile={config.profile}, workers={workers})",
    )

    if workers > 1:
        import os

        os.environ["DIRECTOR_PROFILE"] = profile
        os.environ["DIRECTOR_SERVER_HOST"] = host
        os.environ["DIRECTOR_SERVER_PORT"] = str(port)
        uvicorn.run(
            "director_ai.server:create_app",
            factory=True,
            host=host,
            port=port,
            workers=workers,
        )
    else:
        app = create_app(config)
        uvicorn.run(app, host=host, port=port)


def _cmd_stress_test(args: list[str]) -> None:
    """Benchmark streaming kernel throughput."""
    import math
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    streams = 100
    tokens_per_stream = 50
    concurrency = 8
    json_output = False

    i = 0
    while i < len(args):
        if args[i] == "--streams" and i + 1 < len(args):
            streams = int(args[i + 1])
            i += 2
        elif args[i] == "--tokens-per-stream" and i + 1 < len(args):
            tokens_per_stream = int(args[i + 1])
            i += 2
        elif args[i] == "--concurrency" and i + 1 < len(args):
            concurrency = int(args[i + 1])
            i += 2
        elif args[i] == "--json":
            json_output = True
            i += 1
        else:
            i += 1

    from director_ai.core.runtime.streaming import StreamingKernel

    def _coherence_cb(token):
        h = hash(token) & 0xFFFFFFFF
        return 0.8 + 0.1 * math.sin(h)

    def _run_one(stream_id):
        kernel = StreamingKernel()
        tokens = [f"tok{j}" for j in range(tokens_per_stream)]
        t0 = time.monotonic()
        session = kernel.stream_tokens(tokens, _coherence_cb)
        elapsed = time.monotonic() - t0
        return {
            "halted": session.halted,
            "tokens": session.token_count,
            "elapsed": elapsed,
        }

    latencies: list[float] = []
    halts = 0
    total_tokens = 0

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_one, sid) for sid in range(streams)]
        for future in as_completed(futures):
            result = future.result()
            latencies.append(result["elapsed"])
            if result["halted"]:
                halts += 1
            total_tokens += result["tokens"]
    t_total = max(time.monotonic() - t_start, 1e-9)

    latencies.sort()

    def _pct(sorted_vals, q):
        idx = int(q * (len(sorted_vals) - 1))
        return sorted_vals[idx]

    report = {
        "streams": streams,
        "tokens_per_stream": tokens_per_stream,
        "concurrency": concurrency,
        "total_seconds": round(t_total, 4),
        "streams_per_second": round(streams / t_total, 2),
        "tokens_per_second": round(total_tokens / t_total, 2),
        "halt_rate": round(halts / streams, 4),
        "latency_p50": round(_pct(latencies, 0.5), 6),
        "latency_p95": round(_pct(latencies, 0.95), 6),
        "latency_p99": round(_pct(latencies, 0.99), 6),
    }

    if json_output:
        print(json.dumps(report, indent=2))
    else:
        print(f"Streams:     {report['streams']}")
        print(f"Tokens/s:    {report['tokens_per_second']}")
        print(f"Streams/s:   {report['streams_per_second']}")
        print(f"Halt rate:   {report['halt_rate']:.2%}")
        print(f"Latency p50: {report['latency_p50'] * 1000:.2f}ms")
        print(f"Latency p95: {report['latency_p95'] * 1000:.2f}ms")
        print(f"Latency p99: {report['latency_p99'] * 1000:.2f}ms")
        print(f"Total time:  {report['total_seconds']:.2f}s")


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
            "  report  [--db PATH] [--since TS] [--until TS] [--format md|json]\n"
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


if __name__ == "__main__":
    main()
