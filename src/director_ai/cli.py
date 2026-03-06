# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Command Line Interface
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
CLI entry point for Director-Class AI.

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

import asyncio
import json
import sys


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — dispatches to subcommands."""
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return

    cmd = args[0]
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
        "serve": _cmd_serve,
        "config": _cmd_config,
        "stress-test": _cmd_stress_test,
        "doctor": _cmd_doctor,
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
        "  ingest <file>         Ingest documents into vector store\n"
        "  eval [--dataset D]    Run NLI benchmark suite\n"
        "  bench [--dataset D] [--seed N] [--output F]  Run regression benchmarks\n"
        "  tune <file.jsonl> [--output config.yaml]  Find optimal threshold\n"
        "  serve [--port N] [--workers W]  Start the FastAPI server\n"
        "  stress-test [options] Benchmark streaming kernel throughput\n"
        "  doctor                Check runtime dependencies and readiness\n"
        "  config [--profile X]  Show/set configuration\n"
    )


def _cmd_version(args: list[str]) -> None:
    import director_ai

    print(f"director-ai {director_ai.__version__}")


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
        "            asyncio.run(store.add(line[:20], line))\n"
        "\n"
        "scorer = CoherenceScorer(\n"
        "    threshold=config.coherence_threshold,\n"
        "    ground_truth_store=store,\n"
        "    use_nli=config.use_nli,\n"
        ")\n"
        "\n"
        "approved, score = asyncio.run(scorer.review(\n"
        '    "What color is the sky?", "The sky is blue."\n'
        "))\n"
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
            f"limit {_BATCH_MAX_FILE_SIZE // 1024 // 1024} MB)"
        )
        sys.exit(1)

    from director_ai.core.agent import CoherenceAgent
    from director_ai.core.batch import BatchProcessor

    prompts: list[str] = []
    with open(input_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if len(line) > _BATCH_MAX_LINE_SIZE:
                print(
                    f"Warning: skipping line {line_no} "
                    f"({len(line)} chars > {_BATCH_MAX_LINE_SIZE} limit)"
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
                        }
                    )
                    + "\n"
                )
        print(f"Results written to {output_file}")


_INGEST_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def _cmd_ingest(args: list[str]) -> None:
    """Ingest files or directories into a VectorGroundTruthStore.

    Supported formats: ``.txt``, ``.md`` (paragraph-chunked), ``.jsonl``/``.json``.
    Directories are walked recursively for supported file types.
    """
    if not args:
        print(
            "Usage: director-ai ingest <file-or-dir> "
            "[--persist <dir>] [--chunk-size <tokens>]"
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

    if not os.path.exists(input_path):
        print(f"Error: path not found: {input_path}")
        sys.exit(1)

    from director_ai.core.config import DirectorConfig

    cfg = DirectorConfig.from_env()
    if persist_dir:
        cfg.vector_backend = "chroma"
        cfg.chroma_persist_dir = persist_dir
    store = cfg.build_store()

    supported_exts = {".txt", ".md", ".json", ".jsonl"}

    def _collect_files(path: str) -> list[Path]:
        p = Path(path)
        if p.is_file():
            return [p]
        files: list[Path] = []
        for ext in supported_exts:
            files.extend(p.rglob(f"*{ext}"))
        return sorted(files)

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
        text = path.read_text(encoding="utf-8", errors="replace")
        ext = path.suffix.lower()
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
                    f"Error: --quantize must be 'int8' or 'fp16', got '{quantize_mode}'"
                )
                sys.exit(1)
            i += 2
        else:
            i += 1

    if quantize_mode:
        from director_ai.core.nli import export_onnx

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
            "Run from the director-ai repo root, or install in editable mode."
        )
        sys.exit(1)

    if dataset and dataset == "aggrefact" and not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not set — AggreFact benchmark may be skipped")

    print(
        f"Running benchmarks (max_samples={max_samples}, model={model or 'default'})..."
    )
    results = _run_suite(model, max_samples)
    _print_comparison_table({model if model else "default": results})

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
            "Run from the director-ai repo root, or install in editable mode."
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

    passed = 0
    failed = 0
    results = []
    for test_fn in suite:
        try:
            test_fn()
            passed += 1
            results.append({"test": test_fn.__name__, "status": "passed"})
        except AssertionError as e:
            failed += 1
            results.append(
                {"test": test_fn.__name__, "status": "failed", "error": str(e)}
            )
            print(f"  FAIL: {test_fn.__name__}: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\n  {passed} passed, {failed} failed in {elapsed:.2f}s")

    report = {
        "dataset": dataset,
        "seed": seed,
        "passed": passed,
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

    from director_ai.core.tuner import tune

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
                f"w_fact: {result.w_fact}\n"
            )
        print(f"Config written to {output_file}")


def _cmd_serve(args: list[str]) -> None:
    port = 8080
    host = "0.0.0.0"
    profile = "default"
    workers = 1
    transport = "http"

    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
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
    config.server_host = host
    config.server_port = port

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
        f"(profile={config.profile}, workers={workers})"
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

    from director_ai.core.streaming import StreamingKernel

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
    py_ok = tuple(int(x) for x in py_ver.split(".")[:2]) >= (3, 10)
    checks.append(("Python >= 3.10", py_ok, py_ver))

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
        from director_ai.core.nli import nli_available

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


if __name__ == "__main__":
    main()
