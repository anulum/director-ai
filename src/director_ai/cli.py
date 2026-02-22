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

    cmd = args[0]
    rest = args[1:]

    commands = {
        "version": _cmd_version,
        "review": _cmd_review,
        "process": _cmd_process,
        "batch": _cmd_batch,
        "ingest": _cmd_ingest,
        "serve": _cmd_serve,
        "config": _cmd_config,
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
        "  review <prompt> <resp> Review a prompt/response pair\n"
        "  process <prompt>      Process a prompt through the full pipeline\n"
        "  batch <file.jsonl>    Batch process (max 10K prompts, <100MB)\n"
        "  ingest <file>         Ingest documents into vector store\n"
        "  serve [--port N]      Start the FastAPI server\n"
        "  config [--profile X]  Show/set configuration\n"
    )


def _cmd_version(args: list[str]) -> None:
    import director_ai

    print(f"director-ai {director_ai.__version__}")


def _cmd_review(args: list[str]) -> None:
    if len(args) < 2:
        print("Usage: director-ai review <prompt> <response>")
        sys.exit(1)

    prompt, response = args[0], args[1]

    from director_ai.core.scorer import CoherenceScorer

    scorer = CoherenceScorer(threshold=0.6)
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

    agent = CoherenceAgent()
    result = agent.process(prompt)

    print(f"Output:     {result.output}")
    print(f"Halted:     {result.halted}")
    print(f"Candidates: {result.candidates_evaluated}")
    if result.coherence:
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
        if idx + 1 < len(args):
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

    agent = CoherenceAgent()
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
    """Ingest a text or JSONL file into a VectorGroundTruthStore.

    Supported formats:
      - ``.txt``: one document per line
      - ``.jsonl``: one JSON object per line with a ``"text"`` field
    """
    if not args:
        print("Usage: director-ai ingest <file.txt|file.jsonl> [--persist <dir>]")
        sys.exit(1)

    import os

    input_file = args[0]
    persist_dir = None
    if "--persist" in args:
        idx = args.index("--persist")
        if idx + 1 < len(args):
            persist_dir = args[idx + 1]

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    file_size = os.path.getsize(input_file)
    if file_size > _INGEST_MAX_FILE_SIZE:
        print(
            f"Error: file too large ({file_size / 1024 / 1024:.1f} MB, "
            f"limit {_INGEST_MAX_FILE_SIZE // 1024 // 1024} MB)"
        )
        sys.exit(1)

    from director_ai.core.vector_store import (
        InMemoryBackend,
        VectorBackend,
        VectorGroundTruthStore,
    )

    backend: VectorBackend
    if persist_dir:
        try:
            from director_ai.core.vector_store import ChromaBackend

            backend = ChromaBackend(persist_directory=persist_dir)
        except ImportError:
            print("ChromaDB required for --persist. pip install director-ai[vector]")
            sys.exit(1)
    else:
        backend = InMemoryBackend()

    store = VectorGroundTruthStore(backend=backend, auto_index=False)

    texts: list[str] = []
    is_jsonl = input_file.endswith(".jsonl") or input_file.endswith(".json")

    with open(input_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if is_jsonl:
                try:
                    data = json.loads(line)
                    text = data.get("text", data.get("content", ""))
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping line {line_no}: {e}")
                    continue
            else:
                text = line
            if text:
                texts.append(text)

    count = store.ingest(texts)
    print(f"Ingested {count} documents into vector store.")
    if persist_dir:
        print(f"Persisted to: {persist_dir}")
    else:
        print("(in-memory only — use --persist <dir> to save)")


def _cmd_serve(args: list[str]) -> None:
    port = 8080
    host = "0.0.0.0"
    profile = "default"

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
        else:
            i += 1

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install director-ai[server]")
        sys.exit(1)

    from director_ai.core.config import DirectorConfig
    from director_ai.server import create_app

    if profile != "default":
        config = DirectorConfig.from_profile(profile)
    else:
        config = DirectorConfig.from_env()
    config.server_host = host
    config.server_port = port

    app = create_app(config)
    print(f"Starting Director AI server on {host}:{port} (profile={config.profile})")
    uvicorn.run(app, host=host, port=port)


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
