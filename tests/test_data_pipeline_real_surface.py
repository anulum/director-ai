# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Training Data Pipeline Real-Surface Tests
"""Real CLI coverage for the training data pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from training import data_pipeline

datasets = pytest.importorskip("datasets")

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSON Lines training rows using the production schema."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _balanced_rows() -> list[dict[str, object]]:
    """Return enough rows for the pipeline's stratified 90/10 split."""
    return [
        {
            "premise": f"evidence {index}",
            "hypothesis": f"claim {index}",
            "label": index % 3,
            "source": "local_contract",
        }
        for index in range(30)
    ]


def _training_examples(
    count: int,
    source: str = "contract",
) -> list[data_pipeline.TrainingExample]:
    """Return typed rows for direct production-builder checks."""
    rows: list[data_pipeline.TrainingExample] = []
    for index in range(count):
        rows.append(
            {
                "premise": f"premise {source} {index}",
                "hypothesis": f"hypothesis {source} {index}",
                "label": index % 3,
                "source": source,
            }
        )
    return rows


def _loader(
    rows: list[data_pipeline.TrainingExample],
) -> Callable[[], list[data_pipeline.TrainingExample]]:
    """Return a typed zero-argument loader for orchestration checks."""

    def load() -> list[data_pipeline.TrainingExample]:
        return list(rows)

    return load


def _raising_loader(
    exc: Exception,
) -> Callable[[], list[data_pipeline.TrainingExample]]:
    """Return a typed zero-argument loader that raises an expected exception."""

    def load() -> list[data_pipeline.TrainingExample]:
        raise exc

    return load


def _set_load_dataset(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    """Route Hugging Face loader calls to a typed in-memory dataset object."""

    def load_dataset(*args: object, **kwargs: object) -> object:
        return value

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)


def _label_counts(rows: Iterable[data_pipeline.TrainingExample]) -> dict[int, int]:
    """Return label counts for compact assertions in split-safe fixtures."""
    counts = {0: 0, 1: 0, 2: 0}
    for row in rows:
        counts[row["label"]] += 1
    return counts


def test_data_pipeline_cli_builds_real_local_dataset_pack(tmp_path: Path) -> None:
    """The CLI builds an on-disk DatasetDict from a real local JSONL pack."""
    source = tmp_path / "local_nli.jsonl"
    output = tmp_path / "dataset"
    _write_jsonl(source, _balanced_rows())

    result = subprocess.run(
        [
            sys.executable,
            "training/data_pipeline.py",
            "--local-source-jsonl",
            str(source),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    dataset = datasets.load_from_disk(str(output))
    assert set(dataset) == {"train", "eval"}
    assert dataset["train"].features["label"].names == [
        "entailment",
        "neutral",
        "contradiction",
    ]
    assert len(dataset["train"]) + len(dataset["eval"]) == 30

    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert stats["total"] == 30
    assert stats["source_distribution"] == {"local_contract": 30}
    assert stats["label_distribution"] == {"0": 10, "1": 10, "2": 10}


def test_data_pipeline_cli_rejects_invalid_local_rows(tmp_path: Path) -> None:
    """The CLI fails closed when a local source row has an invalid label."""
    source = tmp_path / "bad_nli.jsonl"
    output = tmp_path / "dataset"
    bad_rows = _balanced_rows()
    bad_rows[3] = {
        "premise": "evidence",
        "hypothesis": "claim",
        "label": "unsupported",
        "source": "local_contract",
    }
    _write_jsonl(source, bad_rows)

    result = subprocess.run(
        [
            sys.executable,
            "training/data_pipeline.py",
            "--local-source-jsonl",
            str(source),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "bad_nli.jsonl:4" in result.stderr
    assert "label must be 0, 1, or 2" in result.stderr
    assert not output.exists()


def test_data_pipeline_main_builds_local_dataset_in_process(tmp_path: Path) -> None:
    """The public CLI entrypoint builds a local pack inside the current process."""
    source = tmp_path / "local_nli.jsonl"
    output = tmp_path / "dataset"
    _write_jsonl(source, _balanced_rows())

    status = data_pipeline.main(
        [
            "--local-source-jsonl",
            str(source),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    dataset = datasets.load_from_disk(str(output))
    assert set(dataset) == {"train", "eval"}
    assert len(dataset["train"]) + len(dataset["eval"]) == 30
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert stats["total"] == 30
    assert stats["source_distribution"] == {"local_contract": 30}


def test_data_pipeline_main_all_keeps_local_sources_air_gapped(
    tmp_path: Path,
) -> None:
    """The --all shortcut still uses only local packs when they are provided."""
    source = tmp_path / "local_nli.jsonl"
    output = tmp_path / "dataset"
    _write_jsonl(source, _balanced_rows())

    status = data_pipeline.main(
        [
            "--all",
            "--local-source-jsonl",
            str(source),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert stats["source_distribution"] == {"local_contract": 30}


def test_data_pipeline_build_dataset_combines_multiple_local_packs(
    tmp_path: Path,
) -> None:
    """The production builder concatenates repeated local JSONL source packs."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "dataset"
    _write_jsonl(first, _balanced_rows())
    _write_jsonl(
        second,
        [
            {
                **row,
                "source": "local_contract_extra",
            }
            for row in _balanced_rows()
        ],
    )

    dataset = data_pipeline.build_dataset(
        local_source_jsonl=[first, second],
        output_dir=output,
    )

    assert len(dataset["train"]) + len(dataset["eval"]) == 60
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert stats["source_distribution"] == {
        "local_contract": 30,
        "local_contract_extra": 30,
    }


@pytest.mark.parametrize(
    ("filename", "lines", "message"),
    [
        ("bad_json.jsonl", ["{"], "bad_json.jsonl:1: invalid JSON"),
        ("array_row.jsonl", ["[]"], "array_row.jsonl:1: row must be a JSON object"),
        (
            "blank_premise.jsonl",
            [
                json.dumps(
                    {
                        "premise": " ",
                        "hypothesis": "claim",
                        "label": 0,
                        "source": "local_contract",
                    }
                )
            ],
            "blank_premise.jsonl:1: premise must be a non-empty string",
        ),
        (
            "bool_label.jsonl",
            [
                json.dumps(
                    {
                        "premise": "evidence",
                        "hypothesis": "claim",
                        "label": True,
                        "source": "local_contract",
                    }
                )
            ],
            "bool_label.jsonl:1: label must be 0, 1, or 2",
        ),
    ],
)
def test_data_pipeline_local_source_validation_errors(
    tmp_path: Path,
    filename: str,
    lines: list[str],
    message: str,
) -> None:
    """Local source packs fail closed on malformed JSONL records."""
    source = tmp_path / filename
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        data_pipeline.build_dataset(
            local_source_jsonl=[source],
            output_dir=tmp_path / "dataset",
        )


def test_data_pipeline_local_source_rejects_missing_file(tmp_path: Path) -> None:
    """The local source path must exist before build work begins."""
    source = tmp_path / "missing.jsonl"

    with pytest.raises(ValueError, match="local source JSONL file does not exist"):
        data_pipeline.build_dataset(
            local_source_jsonl=[source],
            output_dir=tmp_path / "dataset",
        )


def test_data_pipeline_local_source_rejects_empty_file(tmp_path: Path) -> None:
    """Blank-only local packs are rejected before split construction."""
    source = tmp_path / "empty.jsonl"
    source.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no rows"):
        data_pipeline.build_dataset(
            local_source_jsonl=[source],
            output_dir=tmp_path / "dataset",
        )


def test_data_pipeline_local_source_rejects_missing_split_label(
    tmp_path: Path,
) -> None:
    """Local packs must contain every NLI label before stratified splitting."""
    source = tmp_path / "missing_label.jsonl"
    rows = [
        {
            "premise": f"evidence {index}",
            "hypothesis": f"claim {index}",
            "label": index % 2,
            "source": "local_contract",
        }
        for index in range(12)
    ]
    _write_jsonl(source, rows)

    with pytest.raises(ValueError, match="missing labels for stratified split"):
        data_pipeline.build_dataset(
            local_source_jsonl=[source],
            output_dir=tmp_path / "dataset",
        )


def test_data_pipeline_local_source_rejects_sparse_split_label(
    tmp_path: Path,
) -> None:
    """Local packs must have at least two examples per NLI label."""
    source = tmp_path / "sparse_label.jsonl"
    rows = [
        {
            "premise": f"evidence {index}",
            "hypothesis": f"claim {index}",
            "label": label,
            "source": "local_contract",
        }
        for index, label in enumerate([0, 0, 1, 1, 2], start=1)
    ]
    _write_jsonl(source, rows)

    with pytest.raises(ValueError, match="at least two examples"):
        data_pipeline.build_dataset(
            local_source_jsonl=[source],
            output_dir=tmp_path / "dataset",
        )


def test_data_pipeline_main_reports_local_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entrypoint reports local pack validation errors via argparse."""
    source = tmp_path / "bad_nli.jsonl"
    output = tmp_path / "dataset"
    rows = _balanced_rows()
    rows[3] = {
        "premise": "evidence",
        "hypothesis": "claim",
        "label": "unsupported",
        "source": "local_contract",
    }
    _write_jsonl(source, rows)

    with pytest.raises(SystemExit) as exc_info:
        data_pipeline.main(
            [
                "--local-source-jsonl",
                str(source),
                "--output-dir",
                str(output),
            ]
        )

    assert exc_info.value.code == 2
    assert "bad_nli.jsonl:4" in capsys.readouterr().err
    assert not output.exists()


def test_data_pipeline_build_dataset_rejects_empty_remote_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote-source builds fail closed when every loader returns no examples."""
    empty_loader = _loader([])
    monkeypatch.setattr(data_pipeline, "_load_halueval", empty_loader)
    monkeypatch.setattr(data_pipeline, "_load_fever", empty_loader)
    monkeypatch.setattr(data_pipeline, "_load_vitaminc", empty_loader)
    monkeypatch.setattr(data_pipeline, "_load_anli_r3", empty_loader)

    with pytest.raises(ValueError, match="produced no examples"):
        data_pipeline.build_dataset(output_dir=tmp_path / "dataset")


def test_data_pipeline_build_dataset_caps_vitaminc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote-source builds cap VitaminC before combining training sources."""
    monkeypatch.setattr(data_pipeline, "VITAMINC_CAP", 6)
    monkeypatch.setattr(data_pipeline, "_load_halueval", _loader(_training_examples(6)))
    monkeypatch.setattr(data_pipeline, "_load_fever", _loader(_training_examples(6)))
    monkeypatch.setattr(
        data_pipeline,
        "_load_vitaminc",
        _loader(_training_examples(12, "vitaminc")),
    )
    monkeypatch.setattr(data_pipeline, "_load_anli_r3", _loader(_training_examples(6)))

    data_pipeline.build_dataset(output_dir=tmp_path / "dataset")

    stats = json.loads(
        (tmp_path / "dataset" / "stats.json").read_text(encoding="utf-8")
    )
    assert stats["total"] == 24
    assert stats["source_distribution"]["vitaminc"] == 6


def test_data_pipeline_build_dataset_includes_optional_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote-source builds wire RAGTruth, SummaC, and AggreFact toggles."""
    monkeypatch.setattr(data_pipeline, "_load_halueval", _loader(_training_examples(6)))
    monkeypatch.setattr(data_pipeline, "_load_fever", _loader(_training_examples(6)))
    monkeypatch.setattr(data_pipeline, "_load_vitaminc", _loader(_training_examples(6)))
    monkeypatch.setattr(data_pipeline, "_load_anli_r3", _loader(_training_examples(6)))
    monkeypatch.setattr(
        data_pipeline,
        "_load_ragtruth",
        _loader(_training_examples(6, "ragtruth")),
    )
    monkeypatch.setattr(
        data_pipeline,
        "_load_summac",
        _loader(_training_examples(6, "summac")),
    )
    monkeypatch.setattr(
        data_pipeline,
        "_load_aggrefact",
        _loader(_training_examples(6, "aggrefact")),
    )

    data_pipeline.build_dataset(
        include_ragtruth=True,
        include_summac=True,
        include_aggrefact=True,
        output_dir=tmp_path / "dataset",
    )

    stats = json.loads(
        (tmp_path / "dataset" / "stats.json").read_text(encoding="utf-8")
    )
    assert stats["total"] == 42
    assert stats["source_distribution"]["ragtruth"] == 6
    assert stats["source_distribution"]["summac"] == 6
    assert stats["source_distribution"]["aggrefact"] == 6


def test_data_pipeline_build_dataset_tolerates_unavailable_summac(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SummaC remains optional when the remote dataset is unavailable."""
    monkeypatch.setattr(data_pipeline, "_load_halueval", _loader(_training_examples(6)))
    monkeypatch.setattr(data_pipeline, "_load_fever", _loader(_training_examples(6)))
    monkeypatch.setattr(data_pipeline, "_load_vitaminc", _loader(_training_examples(6)))
    monkeypatch.setattr(data_pipeline, "_load_anli_r3", _loader(_training_examples(6)))
    monkeypatch.setattr(
        data_pipeline,
        "_load_summac",
        _raising_loader(RuntimeError("offline")),
    )

    data_pipeline.build_dataset(
        include_summac=True,
        output_dir=tmp_path / "dataset",
    )

    stats = json.loads(
        (tmp_path / "dataset" / "stats.json").read_text(encoding="utf-8")
    )
    assert stats["total"] == 24


def test_data_pipeline_remote_loaders_pin_huggingface_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote loaders pass immutable revisions to Hugging Face datasets."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    common_row = {
        "knowledge": "K",
        "question": "Q",
        "right_answer": "right",
        "hallucinated_answer": "wrong",
        "dialogue_history": "D",
        "right_response": "right",
        "hallucinated_response": "wrong",
        "document": "Doc",
        "right_summary": "right",
        "hallucinated_summary": "wrong",
        "premise": "P",
        "hypothesis": "H",
        "evidence": "E",
        "claim": "C",
        "context": "Ctx",
        "output": "Out",
        "hallucination_labels_processed": "{}",
        "doc": "Doc",
        "dataset": "AggreFact-CNN",
        "label": 0,
    }

    def load_dataset(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        dataset_id = str(args[0])
        if dataset_id == "mteb/summac":
            return {"train": [common_row]}
        return [common_row]

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    data_pipeline._load_halueval()
    data_pipeline._load_fever()
    data_pipeline._load_vitaminc()
    data_pipeline._load_anli_r3()
    data_pipeline._load_ragtruth()
    data_pipeline._load_summac()
    data_pipeline._load_aggrefact()

    revisions = {str(args[0]): kwargs.get("revision") for args, kwargs in calls}
    for dataset_id, revision in data_pipeline.REMOTE_DATASET_REVISIONS.items():
        assert revisions[dataset_id] == revision
        assert len(revision) == 40
    assert revisions["mteb/summac"] == data_pipeline.SUMMAC_DATASET_REVISION


def test_data_pipeline_fever_skips_non_scalar_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FEVER skips rows whose label cannot be normalised."""
    _set_load_dataset(
        monkeypatch,
        [{"premise": "P", "hypothesis": "H", "label": None}],
    )

    assert data_pipeline._load_fever() == []


def test_data_pipeline_vitaminc_skips_untyped_and_unknown_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VitaminC skips non-normalisable labels and incomplete rows."""
    _set_load_dataset(
        monkeypatch,
        [
            {"evidence": "E", "claim": "C", "label": None},
            {"evidence": "E", "claim": "C", "label": "UNKNOWN"},
        ],
    )

    assert data_pipeline._load_vitaminc() == []


def test_data_pipeline_ragtruth_handles_untyped_label_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAGTruth treats unexpected label payloads as no hallucination labels."""
    _set_load_dataset(
        monkeypatch,
        [
            {
                "context": "context",
                "output": "answer",
                "hallucination_labels_processed": 7,
            }
        ],
    )

    examples = data_pipeline._load_ragtruth()

    assert len(examples) == 1
    assert examples[0]["label"] == data_pipeline.LABEL_ENTAILMENT


def test_data_pipeline_summac_loader_maps_supported_and_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SummaC maps real split rows through the shared NLI label contract."""
    _set_load_dataset(
        monkeypatch,
        {
            "train": [
                {
                    "text": "source article",
                    "claim": "supported summary",
                    "label": 1,
                },
                {
                    "document": "source document",
                    "summary": "unsupported summary",
                    "label": 0,
                },
                {
                    "document": "ignored document",
                    "summary": "ignored summary",
                    "label": 99,
                },
            ],
        },
    )

    examples = data_pipeline._load_summac()

    assert _label_counts(examples) == {0: 1, 1: 0, 2: 1}
    assert {example["source"] for example in examples} == {"summac"}


def test_data_pipeline_aggrefact_defaults_dataset_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AggreFact rows without dataset metadata use the stable fallback source."""
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _set_load_dataset(
        monkeypatch,
        [{"doc": "D", "claim": "C", "label": 1, "dataset": ""}],
    )

    examples = data_pipeline._load_aggrefact()

    assert len(examples) == 1
    assert examples[0]["source"] == "aggrefact_aggrefact"
