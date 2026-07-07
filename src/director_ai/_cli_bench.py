# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI Benchmark/Training/Export Commands
"""CLI subcommands for benchmarking, training, and model export.

Extracted from cli.py to reduce module size.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from typing import Any, cast

_TuneFunction = Callable[[list[dict[str, Any]]], Any]
_ConfidenceReportFunction = Callable[[Any], str]
_ProfileOverlayFunction = Callable[..., str]


def _cmd_eval(args: list[str]) -> None:
    """Run NLI benchmark suite.

    Usage::

        director-ai eval --dataset aggrefact --max-samples 100 --output results.json
    """
    if args and args[0] in ("-h", "--help", "help"):
        _print_eval_help()
        return

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
    if args and args[0] in ("-h", "--help", "help"):
        _print_bench_help()
        return

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
    tests: dict[str, list[Callable[[], Any]]] = {
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
    if args and args[0] in ("-h", "--help", "help"):
        _print_tune_help()
        return

    if not args:
        _print_tune_help()
        sys.exit(1)

    import os

    input_file = ""
    output_file = None
    profile = "tuned"
    base_profile = ""

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--dataset" and i + 1 < len(args):
            input_file = args[i + 1]
            i += 2
        elif arg == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        elif arg == "--profile" and i + 1 < len(args):
            base_profile = args[i + 1]
            profile = f"{base_profile}_tuned"
            i += 2
        elif not arg.startswith("-") and not input_file:
            input_file = arg
            i += 1
        else:
            i += 1

    if not input_file:
        print("Error: missing dataset file")
        sys.exit(1)

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    samples: list[dict[str, Any]] = []
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

    tune, format_confidence_report, format_profile_overlay = _load_tuner_functions()

    result = tune(samples)

    print(f"Best threshold: {result.threshold}")
    print(f"Weights:        w_logic={result.w_logic}, w_fact={result.w_fact}")
    print(f"Balanced Acc:   {result.balanced_accuracy:.4f}")
    print(f"Precision:      {result.precision:.4f}")
    print(f"Recall:         {result.recall:.4f}")
    print(f"F1:             {result.f1:.4f}")
    print(f"Samples:        {result.samples}")
    print(format_confidence_report(result))

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(
                format_profile_overlay(
                    result,
                    profile=profile,
                    base_profile=base_profile,
                ),
            )
        print(f"Profile overlay written to {output_file}")


def _load_tuner_functions() -> tuple[
    _TuneFunction,
    _ConfidenceReportFunction,
    _ProfileOverlayFunction,
]:
    """Load threshold-tuner functions from the canonical calibration module."""
    canonical_tuner = importlib.import_module("director_ai.core.calibration.tuner")

    legacy_functions: (
        tuple[
            _TuneFunction,
            _ConfidenceReportFunction,
            _ProfileOverlayFunction,
        ]
        | None
    ) = None
    legacy = sys.modules.get("director_ai.core.training.tuner")
    if legacy is not None and legacy is not canonical_tuner:
        legacy_functions = _tuner_functions_from(legacy)
        if legacy_functions is not None and _prefer_legacy_tuner_override(
            legacy,
            canonical_tuner,
        ):
            return legacy_functions

    canonical_functions = _tuner_functions_from(canonical_tuner)
    if canonical_functions is None:
        raise RuntimeError("canonical calibration tuner lacks required functions")
    return canonical_functions


def _tuner_functions_from(
    module: object,
) -> tuple[_TuneFunction, _ConfidenceReportFunction, _ProfileOverlayFunction] | None:
    """Return tuner callables from a module-shaped object when all are present."""
    tune = getattr(module, "tune", None)
    confidence_report = getattr(module, "format_confidence_report", None)
    profile_overlay = getattr(module, "format_profile_overlay", None)
    if callable(tune) and callable(confidence_report) and callable(profile_overlay):
        return (
            cast("_TuneFunction", tune),
            cast("_ConfidenceReportFunction", confidence_report),
            cast("_ProfileOverlayFunction", profile_overlay),
        )
    return None


def _prefer_legacy_tuner_override(legacy: object, canonical: object) -> bool:
    """Return whether a distinct legacy module should override canonical import."""
    canonical_file = getattr(canonical, "__file__", None)
    legacy_file = getattr(legacy, "__file__", None)
    return isinstance(canonical_file, str) and legacy_file != canonical_file


def _print_eval_help() -> None:
    """Print eval subcommand options without loading benchmark modules."""
    print(
        "Usage: director-ai eval [options]\n"
        "\n"
        "Run the NLI benchmark suite or export a quantized ONNX scorer.\n"
        "\n"
        "Options:\n"
        "  --dataset NAME          Benchmark dataset, for example aggrefact\n"
        "  --max-samples N         Limit benchmark samples\n"
        "  --output FILE           Write benchmark results as JSON\n"
        "  --model ID              Override scorer model identifier\n"
        "  --quantize int8|fp16    Export an ONNX scorer with quantization\n"
    )


def _print_bench_help() -> None:
    """Print bench subcommand options without running benchmark functions."""
    print(
        "Usage: director-ai bench [options]\n"
        "\n"
        "Run the repository regression benchmark suite.\n"
        "\n"
        "Options:\n"
        "  --dataset regression|e2e|streaming\n"
        "  --seed N                Seed deterministic benchmark ordering\n"
        "  --max-samples N         Limit the selected benchmark tests\n"
        "  --output FILE           Write benchmark report as JSON\n"
    )


def _print_tune_help() -> None:
    """Print tune subcommand options without opening a dataset file."""
    print(
        "Usage: director-ai tune <labeled.jsonl>|--dataset <eval.jsonl> [options]\n"
        "\n"
        "Tune profile thresholds from labelled prompt/response samples.\n"
        "\n"
        "Options:\n"
        "  --dataset FILE          Labelled evaluation JSONL input\n"
        "  --profile NAME          Base profile name for the overlay\n"
        "  --output FILE           Write tuned profile YAML\n"
    )


def _print_finetune_help() -> None:
    """Print finetune subcommand options without opening training data."""
    print(
        "Usage: director-ai finetune <train.jsonl> [options]\n"
        "\n"
        "Fine-tune the NLI scorer on domain-specific labelled data.\n"
        "\n"
        "Options:\n"
        "  --eval <eval.jsonl>     Evaluation data\n"
        "  --output <dir>          Output directory (default: ./director-finetuned)\n"
        "  --epochs N              Training epochs (default: 3)\n"
        "  --lr FLOAT              Learning rate (default: 2e-5)\n"
        "  --batch-size N          Batch size (default: 16)\n"
        "  --base-model ID         Base model (default: FactCG-DeBERTa-v3-Large)\n"
        "  --mix-general           Mix general NLI data to prevent forgetting\n"
        "  --general-data FILE     General NLI JSONL for --mix-general\n"
        "  --early-stopping N      Stop after N evals without improvement\n"
        "  --class-weights         Apply inverse-frequency class weights\n"
        "  --auto-benchmark        Run anti-regression check after training\n"
        "  --auto-onnx             Export to ONNX after training\n"
    )


def _cmd_finetune(args: list[str]) -> None:
    """Fine-tune NLI model on domain-specific labeled data.

    Input JSONL format: ``{"premise": str, "hypothesis": str, "label": int}``
    Also accepts: ``{"doc": str, "claim": str, "label": int}``
    """
    if args and args[0] in ("-h", "--help", "help"):
        _print_finetune_help()
        return

    if not args:
        _print_finetune_help()
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
            f"  Regression:      {rr['regression_pp']:+.1f}pp → {rr['recommendation']}",
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
