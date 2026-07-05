# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Training Data Pipeline
"""
Build unified NLI training dataset from seven sources:

- HaluEval (QA + Dialogue + Summarization) → ~60K
- FEVER (claims + evidence) → ~203K
- VitaminC (claims + evidence, capped) → ~100K
- ANLI Round 3 (hardest NLI split) → ~100K
- RAGTruth (RAG hallucination labels) → ~variable
- SummaC (summarisation consistency) → ~variable
- LLM-AggreFact (11 sub-datasets, gated) → ~29K

All normalised to (premise, hypothesis, label) with 3-class labels:
    0 = entailment, 1 = neutral, 2 = contradiction.

Usage::

    python training/data_pipeline.py
    python training/data_pipeline.py --include-ragtruth --include-summac
    python training/data_pipeline.py --include-aggrefact  # requires HF_TOKEN
    python training/data_pipeline.py --local-source-jsonl evidence/local_nli.jsonl
    python training/data_pipeline.py --all  # all sources
    # Output: training/data/ (HuggingFace Dataset on disk)
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from datasets import DatasetDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "data"

LABEL_ENTAILMENT = 0
LABEL_NEUTRAL = 1
LABEL_CONTRADICTION = 2
VALID_LABELS = frozenset({LABEL_ENTAILMENT, LABEL_NEUTRAL, LABEL_CONTRADICTION})
REMOTE_DATASET_REVISIONS = {
    "pminervini/HaluEval": "12a856119f03975a94509091e8cada3e6be6ead7",
    "pietrolesci/nli_fever": "1eddac63112eee1fdf1966e0bca27a5ff248c772",
    "tals/vitaminc": "be6febb761b0b2807687e61e0b5282e459df2fa0",
    "anli": "8e4813d81f46d313dac7892e1c28076917cfcdf9",
    "wandb/RAGTruth-processed": "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b",
    "lytang/LLM-AggreFact": "981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4",
}
SUMMAC_DATASET_REVISION = (
    os.environ.get("DIRECTOR_AI_SUMMAC_REVISION", "").strip() or "main"
)


class TrainingExample(TypedDict):
    """Normalised NLI row consumed by the training data pipeline."""

    premise: str
    hypothesis: str
    label: int
    source: str


def _as_rows(dataset: object) -> Iterable[Mapping[str, object]]:
    """Return a typed iterator over Hugging Face dataset rows."""
    return cast(Iterable[Mapping[str, object]], dataset)


def _string_field(row: Mapping[str, object], key: str) -> str:
    """Return a string field from a heterogeneous dataset row."""
    value = row.get(key, "")
    return value if isinstance(value, str) else ""


def _load_halueval() -> list[TrainingExample]:
    """Load HaluEval QA + Dialogue + Summarization from HuggingFace."""
    from datasets import load_dataset

    examples: list[TrainingExample] = []

    for task in ("qa", "dialogue", "summarization"):
        logger.info("Loading HaluEval/%s ...", task)
        ds = load_dataset(
            "pminervini/HaluEval",
            task,
            split="data",
            revision=REMOTE_DATASET_REVISIONS["pminervini/HaluEval"],
        )

        for row in _as_rows(ds):
            if task == "qa":
                premise = _string_field(row, "knowledge") or _string_field(
                    row, "question"
                )
                right = _string_field(row, "right_answer")
                halluc = _string_field(row, "hallucinated_answer")
            elif task == "dialogue":
                premise = _string_field(row, "dialogue_history") or _string_field(
                    row, "knowledge"
                )
                right = _string_field(row, "right_response")
                halluc = _string_field(row, "hallucinated_response")
            else:
                premise = _string_field(row, "document")
                right = _string_field(row, "right_summary")
                halluc = _string_field(row, "hallucinated_summary")

            if premise and right:
                examples.append(
                    {
                        "premise": premise,
                        "hypothesis": right,
                        "label": LABEL_ENTAILMENT,
                        "source": f"halueval_{task}",
                    }
                )
            if premise and halluc:
                examples.append(
                    {
                        "premise": premise,
                        "hypothesis": halluc,
                        "label": LABEL_CONTRADICTION,
                        "source": f"halueval_{task}",
                    }
                )

    logger.info("HaluEval: %d examples", len(examples))
    return examples


def _load_fever() -> list[TrainingExample]:
    """Load FEVER dataset (claims + evidence + labels)."""
    from datasets import load_dataset

    logger.info("Loading FEVER ...")
    ds = load_dataset(
        "pietrolesci/nli_fever",
        split="train",
        revision=REMOTE_DATASET_REVISIONS["pietrolesci/nli_fever"],
    )

    label_map = {
        "entailment": LABEL_ENTAILMENT,
        "neutral": LABEL_NEUTRAL,
        "contradiction": LABEL_CONTRADICTION,
    }

    examples: list[TrainingExample] = []
    for row in _as_rows(ds):
        premise = _string_field(row, "premise")
        hypothesis = _string_field(row, "hypothesis")
        raw_label = row.get("label")

        label: int | None
        if isinstance(raw_label, int):
            label = raw_label
        elif isinstance(raw_label, str):
            label = label_map.get(raw_label.lower())
        else:
            continue

        if label is None or not premise or not hypothesis:
            continue

        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "source": "fever",
            }
        )

    logger.info("FEVER: %d examples", len(examples))
    return examples


def _load_vitaminc() -> list[TrainingExample]:
    """Load VitaminC (fact verification with evidence)."""
    from datasets import load_dataset

    logger.info("Loading VitaminC ...")
    ds = load_dataset(
        "tals/vitaminc",
        split="train",
        revision=REMOTE_DATASET_REVISIONS["tals/vitaminc"],
    )

    label_map = {
        "SUPPORTS": LABEL_ENTAILMENT,
        "REFUTES": LABEL_CONTRADICTION,
        "NOT ENOUGH INFO": LABEL_NEUTRAL,
    }

    examples: list[TrainingExample] = []
    for row in _as_rows(ds):
        premise = _string_field(row, "evidence")
        hypothesis = _string_field(row, "claim")
        raw_label = row.get("label")

        label: int | None
        if isinstance(raw_label, int):
            label = raw_label
        elif isinstance(raw_label, str):
            label = label_map.get(raw_label.upper())
        else:
            continue

        if label is None or not premise or not hypothesis:
            continue

        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "source": "vitaminc",
            }
        )

    logger.info("VitaminC: %d examples", len(examples))
    return examples


def _load_anli_r3() -> list[TrainingExample]:
    """Load ANLI Round 3 (hardest adversarial NLI split)."""
    from datasets import load_dataset

    logger.info("Loading ANLI Round 3 ...")
    ds = load_dataset(
        "anli",
        split="train_r3",
        revision=REMOTE_DATASET_REVISIONS["anli"],
    )

    examples: list[TrainingExample] = []
    for row in _as_rows(ds):
        premise = _string_field(row, "premise")
        hypothesis = _string_field(row, "hypothesis")
        label = row.get("label")

        if not isinstance(label, int) or not premise or not hypothesis:
            continue

        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": int(label),
                "source": "anli_r3",
            }
        )

    logger.info("ANLI R3: %d examples", len(examples))
    return examples


def _load_ragtruth() -> list[TrainingExample]:
    """Load RAGTruth (RAG hallucination detection) from HuggingFace.

    Uses wandb/RAGTruth-processed with per-response hallucination labels.
    Maps: no hallucination → entailment, hallucination → contradiction.
    """
    from datasets import load_dataset

    logger.info("Loading RAGTruth (wandb/RAGTruth-processed) ...")
    ds = load_dataset(
        "wandb/RAGTruth-processed",
        split="test",
        revision=REMOTE_DATASET_REVISIONS["wandb/RAGTruth-processed"],
    )

    examples: list[TrainingExample] = []
    for row in _as_rows(ds):
        context = _string_field(row, "context")
        response = _string_field(row, "output")
        if not context or not response:
            continue

        labels_raw = row.get("hallucination_labels_processed", "{}")
        if isinstance(labels_raw, str):
            labels = cast(Mapping[str, object], ast.literal_eval(labels_raw))
        elif isinstance(labels_raw, Mapping):
            labels = labels_raw
        else:
            labels = {}

        evident_conflict = labels.get("evident_conflict", 0)
        baseless_info = labels.get("baseless_info", 0)
        is_hallucinated = (
            isinstance(evident_conflict, int) and evident_conflict > 0
        ) or (isinstance(baseless_info, int) and baseless_info > 0)

        examples.append(
            {
                "premise": context[:2000],
                "hypothesis": response[:2000],
                "label": LABEL_CONTRADICTION if is_hallucinated else LABEL_ENTAILMENT,
                "source": "ragtruth",
            }
        )

    logger.info("RAGTruth: %d examples", len(examples))
    return examples


def _load_summac() -> list[TrainingExample]:
    """Load SummaC (summarisation consistency) from HuggingFace.

    Binary labels: 1 = consistent (entailment), 0 = inconsistent (contradiction).
    """
    from datasets import load_dataset

    logger.info("Loading SummaC (mteb/summac) ...")
    ds = load_dataset("mteb/summac", revision=SUMMAC_DATASET_REVISION)

    label_map = {1: LABEL_ENTAILMENT, 0: LABEL_CONTRADICTION}

    examples: list[TrainingExample] = []
    for split_name in cast(Iterable[str], ds):
        split_rows = cast(object, ds[split_name])
        for row in _as_rows(split_rows):
            doc = _string_field(row, "text") or _string_field(row, "document")
            claim = _string_field(row, "claim") or _string_field(row, "summary")
            label = row.get("label")

            if not doc or not claim or label not in label_map:
                continue

            examples.append(
                {
                    "premise": doc[:2000],
                    "hypothesis": claim,
                    "label": label_map[label],
                    "source": "summac",
                }
            )

    logger.info("SummaC: %d examples", len(examples))
    return examples


def _load_aggrefact() -> list[TrainingExample]:
    """Load LLM-AggreFact (gated, requires HF_TOKEN).

    11 sub-datasets: summarisation, RAG, and grounding tasks.
    Binary labels: 1 = supported (entailment), 0 = not supported (contradiction).
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — skipping LLM-AggreFact (gated dataset)")
        return []

    logger.info("Loading LLM-AggreFact (lytang/LLM-AggreFact) ...")
    ds = load_dataset(
        "lytang/LLM-AggreFact",
        split="test",
        token=token,
        revision=REMOTE_DATASET_REVISIONS["lytang/LLM-AggreFact"],
    )

    label_map = {1: LABEL_ENTAILMENT, 0: LABEL_CONTRADICTION}

    examples: list[TrainingExample] = []
    for row in _as_rows(ds):
        doc = _string_field(row, "doc")
        claim = _string_field(row, "claim")
        label = row.get("label")
        ds_name = _string_field(row, "dataset") or "aggrefact"

        if not doc or not claim or label not in label_map:
            continue

        examples.append(
            {
                "premise": doc[:2000],
                "hypothesis": claim,
                "label": label_map[label],
                "source": f"aggrefact_{ds_name}",
            }
        )

    logger.info("LLM-AggreFact: %d examples", len(examples))
    return examples


VITAMINC_CAP = 100_000  # cap VitaminC to ~30% of total (was 50.6%)


def _required_string(
    path: Path,
    line_number: int,
    row: Mapping[str, object],
    field: str,
) -> str:
    """Return a non-empty string field from a local source row."""
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{path.name}:{line_number}: {field} must be a non-empty string"
        )
    return value


def _required_label(path: Path, line_number: int, row: Mapping[str, object]) -> int:
    """Return a valid 3-class NLI label from a local source row."""
    value = row.get("label")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in VALID_LABELS
    ):
        raise ValueError(f"{path.name}:{line_number}: label must be 0, 1, or 2")
    return value


def _normalise_local_row(
    path: Path,
    line_number: int,
    row: Mapping[str, object],
) -> TrainingExample:
    """Validate and normalise one local JSONL row."""
    return {
        "premise": _required_string(path, line_number, row, "premise"),
        "hypothesis": _required_string(path, line_number, row, "hypothesis"),
        "label": _required_label(path, line_number, row),
        "source": _required_string(path, line_number, row, "source"),
    }


def _load_local_jsonl(path: Path) -> list[TrainingExample]:
    """Load validated local NLI rows from a JSON Lines source pack.

    Parameters
    ----------
    path:
        JSON Lines file where each row contains ``premise``, ``hypothesis``,
        ``label`` (0, 1, 2), and ``source`` fields.

    Returns
    -------
    list[TrainingExample]
        Validated rows ready for the shared DatasetDict construction path.

    Raises
    ------
    ValueError
        If the file is missing, empty, malformed, or contains invalid rows.
    """
    if not path.is_file():
        raise ValueError(f"{path}: local source JSONL file does not exist")

    examples: list[TrainingExample] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{path.name}:{line_number}: row must be a JSON object")
        examples.append(
            _normalise_local_row(
                path,
                line_number,
                cast(Mapping[str, object], loaded),
            )
        )

    if not examples:
        raise ValueError(f"{path.name}: local source JSONL file contains no rows")
    logger.info("Local source %s: %d examples", path, len(examples))
    return examples


def _load_local_sources(paths: Sequence[Path]) -> list[TrainingExample]:
    """Load and concatenate one or more local JSONL source packs."""
    examples: list[TrainingExample] = []
    for path in paths:
        examples.extend(_load_local_jsonl(path))
    return examples


def _validate_examples_for_split(examples: Sequence[TrainingExample]) -> None:
    """Reject empty or split-unsafe example sets before Dataset construction."""
    if not examples:
        raise ValueError("data pipeline produced no examples")

    label_counts = Counter(example["label"] for example in examples)
    missing = sorted(VALID_LABELS - set(label_counts))
    if missing:
        raise ValueError(f"missing labels for stratified split: {missing}")
    sparse = {
        label: count
        for label, count in label_counts.items()
        if label in VALID_LABELS and count < 2
    }
    if sparse:
        raise ValueError(
            f"labels need at least two examples for stratified split: {sparse}"
        )


def build_dataset(
    include_ragtruth: bool = False,
    include_summac: bool = False,
    include_aggrefact: bool = False,
    local_source_jsonl: Sequence[Path] = (),
    output_dir: Path | None = None,
) -> DatasetDict:
    """Build a unified training dataset from remote or local NLI sources.

    Parameters
    ----------
    include_ragtruth:
        Include the optional RAGTruth remote source.
    include_summac:
        Include the optional SummaC remote source.
    include_aggrefact:
        Include the gated LLM-AggreFact remote source when ``HF_TOKEN`` exists.
    local_source_jsonl:
        Optional local JSON Lines source packs. When supplied, these packs are
        used instead of the remote Hugging Face sources so air-gapped and
        regression builds stay deterministic.
    output_dir:
        Directory where the DatasetDict and ``stats.json`` are written.
        Defaults to the module-level ``OUTPUT_DIR``.

    Returns
    -------
    DatasetDict
        Saved training/evaluation split with ``premise``, ``hypothesis``,
        ``label``, and ``source`` fields.
    """
    from datasets import Dataset, DatasetDict

    if local_source_jsonl:
        all_examples = _load_local_sources(local_source_jsonl)
    else:
        halueval = _load_halueval()
        fever = _load_fever()
        vitaminc = _load_vitaminc()
        anli = _load_anli_r3()

        # Cap VitaminC to prevent it dominating training (~370K → 100K)
        if len(vitaminc) > VITAMINC_CAP:
            import random

            # Deterministic data capping; not used for secrets or tokens.
            sampler = random.Random(42)  # nosec B311
            vitaminc = sampler.sample(vitaminc, VITAMINC_CAP)
            logger.info("VitaminC capped to %d examples", VITAMINC_CAP)

        all_examples = halueval + fever + vitaminc + anli

        if include_ragtruth:
            ragtruth = _load_ragtruth()
            all_examples += ragtruth

        if include_summac:
            try:
                summac = _load_summac()
                all_examples += summac
            except Exception as exc:
                logger.warning("SummaC loading failed (may be unavailable): %s", exc)

        if include_aggrefact:
            aggrefact = _load_aggrefact()
            all_examples += aggrefact

    logger.info("Total examples: %d", len(all_examples))
    _validate_examples_for_split(all_examples)
    resolved_output_dir = OUTPUT_DIR if output_dir is None else output_dir

    ds = Dataset.from_list(all_examples)

    # Cast label to ClassLabel so stratified split works
    from datasets import ClassLabel

    ds = ds.cast_column(
        "label", ClassLabel(names=["entailment", "neutral", "contradiction"])
    )

    # Stratified 90/10 split by label
    split = ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    dataset = DatasetDict({"train": split["train"], "eval": split["test"]})

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(resolved_output_dir))
    logger.info("Saved to %s", resolved_output_dir)

    # Stats
    stats = {
        "total": len(all_examples),
        "train": len(dataset["train"]),
        "eval": len(dataset["eval"]),
        "label_distribution": dict(Counter(ex["label"] for ex in all_examples)),
        "source_distribution": dict(Counter(ex["source"] for ex in all_examples)),
    }
    stats_path = resolved_output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Stats: %s", json.dumps(stats, indent=2))

    return dataset


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the data pipeline."""
    parser = argparse.ArgumentParser(description="Build unified NLI training dataset")
    parser.add_argument(
        "--include-ragtruth", action="store_true", help="Include RAGTruth dataset"
    )
    parser.add_argument(
        "--include-summac", action="store_true", help="Include SummaC dataset"
    )
    parser.add_argument(
        "--include-aggrefact",
        action="store_true",
        help="Include LLM-AggreFact (requires HF_TOKEN)",
    )
    parser.add_argument(
        "--local-source-jsonl",
        action="append",
        type=Path,
        default=[],
        help=(
            "Local JSONL source pack with premise, hypothesis, label, and source "
            "fields. Repeat to combine packs. When present, remote sources are "
            "not loaded."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the DatasetDict and stats.json are written.",
    )
    parser.add_argument("--all", action="store_true", help="Include all sources")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data-pipeline CLI and return a process status code."""
    parser = _parser()
    args = parser.parse_args(argv)

    if args.all:
        args.include_ragtruth = True
        args.include_summac = True
        args.include_aggrefact = True

    try:
        build_dataset(
            include_ragtruth=args.include_ragtruth,
            include_summac=args.include_summac,
            include_aggrefact=args.include_aggrefact,
            local_source_jsonl=tuple(args.local_source_jsonl),
            output_dir=args.output_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
