# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — tests for tools/prepare_threshold_data.py

"""Multi-angle coverage for the Julia-tuner feeder: label coercion,
score coercion, file-format dispatch, dropped-row accounting, and end
to-end JSONL emission."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

# ``tools/prepare_threshold_data.py`` is not an installed module,
# so load it by path. ``importlib.util`` keeps imports at the top
# of the file and avoids the ``sys.path.insert`` / E402 anti-pattern.
_TOOLS_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "prepare_threshold_data.py"
)
_spec = importlib.util.spec_from_file_location("prepare_threshold_data", _TOOLS_PATH)
assert _spec is not None and _spec.loader is not None
prepare_threshold_data = importlib.util.module_from_spec(_spec)
sys.modules["prepare_threshold_data"] = prepare_threshold_data
_spec.loader.exec_module(prepare_threshold_data)

_coerce_label = prepare_threshold_data._coerce_label
_coerce_score = prepare_threshold_data._coerce_score
iter_records = prepare_threshold_data.iter_records
main = prepare_threshold_data.main
normalise = prepare_threshold_data.normalise
write_jsonl = prepare_threshold_data.write_jsonl


class TestCoerceLabel:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            (1.0, True),
            (0.0, False),
            ("true", True),
            ("FALSE", False),
            ("SUPPORTED", True),
            ("unsupported", False),
            ("grounded", True),
            ("hallucinated", False),
            ("yes", True),
            ("no", False),
        ],
    )
    def test_accepts_known_values(self, value, expected):
        assert _coerce_label(value) is expected

    @pytest.mark.parametrize("value", [None, "maybe", "", 3.14 + 1j, object()])
    def test_rejects_unknown_values(self, value):
        assert _coerce_label(value) is None

    def test_nan_float_returns_none(self):
        assert _coerce_label(float("nan")) is None


class TestCoerceScore:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.5, 0.5),
            ("0.25", 0.25),
            (1, 1.0),
            (0, 0.0),
        ],
    )
    def test_accepts_numeric(self, value, expected):
        assert _coerce_score(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [None, "", "n/a", object()])
    def test_rejects_non_numeric(self, value):
        assert _coerce_score(value) is None

    def test_rejects_nan(self):
        assert _coerce_score(float("nan")) is None

    def test_rejects_inf(self):
        assert _coerce_score(math.inf) is None
        assert _coerce_score(-math.inf) is None


class TestIterRecords:
    def test_jsonl(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text(
            json.dumps({"score": 0.5, "label": True})
            + "\n"
            + "\n"  # blank line
            + json.dumps({"score": 0.1, "label": False})
            + "\n",
            encoding="utf-8",
        )
        rows = list(iter_records(p))
        assert rows == [
            {"score": 0.5, "label": True},
            {"score": 0.1, "label": False},
        ]

    def test_json_list(self, tmp_path):
        p = tmp_path / "f.json"
        p.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
        assert list(iter_records(p)) == [{"a": 1}, {"a": 2}]

    def test_json_with_records_key(self, tmp_path):
        p = tmp_path / "f.json"
        p.write_text(
            json.dumps({"records": [{"a": 1}]}),
            encoding="utf-8",
        )
        assert list(iter_records(p)) == [{"a": 1}]

    def test_csv(self, tmp_path):
        p = tmp_path / "f.csv"
        p.write_text("score,label\n0.9,true\n0.2,false\n", encoding="utf-8")
        rows = list(iter_records(p))
        assert rows == [
            {"score": "0.9", "label": "true"},
            {"score": "0.2", "label": "false"},
        ]

    def test_rejects_unsupported_extension(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("nope")
        with pytest.raises(ValueError, match="Unsupported extension"):
            list(iter_records(p))

    def test_rejects_malformed_json_doc(self, tmp_path):
        p = tmp_path / "f.json"
        p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="list or a dict"):
            list(iter_records(p))


class TestNormalise:
    def test_drops_records_missing_fields(self):
        rows = [
            {"score": 0.5, "label": True},
            {"score": None, "label": False},
            {"score": 0.3, "label": None},
            {"score": 0.8, "label": "support"},
        ]
        result = list(
            normalise(rows, score_key="score", label_key="label", source_key=None)
        )
        assert result == [
            {"score": 0.5, "label": True},
            {"score": 0.8, "label": True},
        ]

    def test_copies_source_when_present(self):
        rows = [{"score": 0.5, "label": True, "dataset": "summ"}]
        result = list(
            normalise(rows, score_key="score", label_key="label", source_key="dataset")
        )
        assert result == [{"score": 0.5, "label": True, "source": "summ"}]

    def test_ignores_source_when_not_requested(self):
        rows = [{"score": 0.5, "label": True, "dataset": "summ"}]
        result = list(
            normalise(rows, score_key="score", label_key="label", source_key=None)
        )
        assert "source" not in result[0]

    def test_raises_when_no_usable_records(self):
        rows = [{"score": None, "label": None}]
        with pytest.raises(ValueError, match="no usable records"):
            list(normalise(rows, score_key="score", label_key="label", source_key=None))


class TestWriteJsonl:
    def test_writes_one_record_per_line(self, tmp_path):
        out = tmp_path / "sub" / "f.jsonl"
        n = write_jsonl(
            [{"score": 0.5, "label": True}, {"score": 0.1, "label": False}],
            out,
        )
        assert n == 2
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(line) for line in lines] == [
            {"score": 0.5, "label": True},
            {"score": 0.1, "label": False},
        ]


class TestMainCLI:
    def test_end_to_end_csv_to_jsonl(self, tmp_path):
        src = tmp_path / "input.csv"
        src.write_text(
            "score,label,dataset\n0.9,true,summ\n0.2,false,summ\n",
            encoding="utf-8",
        )
        dst = tmp_path / "out.jsonl"
        rc = main(["-i", str(src), "-o", str(dst)])
        assert rc == 0
        lines = dst.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first == {"score": 0.9, "label": True, "source": "summ"}

    def test_returns_nonzero_when_no_output(self, tmp_path):
        src = tmp_path / "input.jsonl"
        src.write_text(
            json.dumps({"score": None, "label": None}) + "\n",
            encoding="utf-8",
        )
        dst = tmp_path / "out.jsonl"
        with pytest.raises(ValueError):
            main(["-i", str(src), "-o", str(dst)])

    def test_custom_keys(self, tmp_path):
        src = tmp_path / "input.jsonl"
        src.write_text(
            json.dumps({"prob": 0.7, "gold": "supported"}) + "\n",
            encoding="utf-8",
        )
        dst = tmp_path / "out.jsonl"
        rc = main(
            [
                "-i",
                str(src),
                "-o",
                str(dst),
                "--score-key",
                "prob",
                "--label-key",
                "gold",
                "--source-key",
                "",
            ]
        )
        assert rc == 0
        row = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
        assert row == {"score": 0.7, "label": True}
