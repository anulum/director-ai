# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the notebook-types canary run
"""Run the notebook-types canaries through the profile's own analysis.

The analysis refuses every interpreter but the pinned build, so the runs that
reach it go through ``DIRECTOR_QUALITY_PYTHON``; the checks made before it,
and the refusal of the lane's own interpreter, run in process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.quality_ratchet_python_support import quality_python
from tests.test_quality_ratchet_typing_cli import commit_typing_sources
from tests.test_quality_ratchet_typing_cli import typing_repository as typing_repository
from tests.test_quality_ratchet_walk import commit, git
from tests.test_quality_ratchet_walk import repo as repo
from tools.quality_ratchet_baseline import BASELINE_PATH
from tools.quality_ratchet_canaries import CANARY_ROOT, fixture_digest, fixture_paths
from tools.quality_ratchet_inventory import python_sources
from tools.quality_ratchet_notebook_canary import (
    PROJECTION_RULES,
    _advisory_keys,
    _last_statement,
    enabled_rules,
    notebook_canary_findings,
)
from tools.quality_ratchet_python_engine import PYTHON_ARTEFACTS
from tools.quality_ratchet_succession import FindingKey

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = f"{CANARY_ROOT}/notebook-types"
PATHS = tuple(
    f"{FIXTURE}/{name}.ipynb.json"
    for name in (
        "canary",
        "refused-assigned-escape",
        "refused-double-escape",
        "refused-escape",
    )
)
BASELINE = json.loads((ROOT / BASELINE_PATH).read_text(encoding="utf-8"))
CODES = sorted(
    {finding["rule"] for finding in BASELINE["profiles"]["notebook-types"]["findings"]}
)


def run_quality(code: str, *arguments: str) -> object:
    """Run Python code under the pinned interpreter and decode its JSON output.

    Parameters
    ----------
    code : str
        The program, which prints one JSON document.
    *arguments : str
        Its arguments.

    Returns
    -------
    object
        The decoded output.
    """
    result = subprocess.run(
        [quality_python(), "-B", "-c", code, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def notebook(*cells: str | list[str], language: str = "python") -> dict[str, object]:
    """Return a notebook holding one Markdown cell and the given code cells.

    Parameters
    ----------
    *cells : str | list[str]
        The code cells' sources, in either nbformat encoding.
    language : str
        The kernel language.

    Returns
    -------
    dict[str, object]
        The notebook.
    """
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"language": language, "name": "python3"},
            "language_info": {"name": language},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": "# Canary"},
            *(
                {"cell_type": "code", "metadata": {}, "outputs": [], "source": cell}
                for cell in cells
            ),
        ],
    }


def test_the_fixture_is_four_notebooks_no_profile_reads(repo: Path) -> None:
    """Keep the fixture out of every inventory and every committed baseline."""
    assert fixture_paths(ROOT, "notebook-types") == PATHS
    assert not [path for path in python_sources(ROOT) if path.startswith(CANARY_ROOT)]
    for name, profile in BASELINE["profiles"].items():
        listed = [*profile["sources"], *(f["path"] for f in profile["findings"])]
        assert not [path for path in listed if path.startswith(CANARY_ROOT)], name
    commit(repo, {path: (ROOT / path).read_bytes() for path in PATHS}, "fixture")
    assert fixture_digest(repo, "notebook-types") == fixture_digest(
        ROOT, "notebook-types"
    )


def test_the_rules_come_from_the_trusted_base_baseline(repo: Path) -> None:
    """Derive the codes from the base commit, never from a later candidate."""
    first = commit(repo, {BASELINE_PATH: BASELINE}, "baseline")
    assert enabled_rules(repo, first) == tuple(sorted({*CODES, *PROJECTION_RULES}))
    fewer = json.loads(json.dumps(BASELINE))
    fewer["profiles"]["notebook-types"]["findings"] = []
    later = commit(repo, {BASELINE_PATH: fewer}, "drop every finding")
    assert enabled_rules(repo, first) == tuple(sorted({*CODES, *PROJECTION_RULES}))
    assert enabled_rules(repo, later) == PROJECTION_RULES
    del fewer["profiles"]["notebook-types"]
    without = commit(repo, {BASELINE_PATH: fewer}, "drop the profile")
    with pytest.raises(ValueError, match="no notebook-types baseline"):
        enabled_rules(repo, without)
    absent = commit(repo, {BASELINE_PATH: None}, "drop the baseline")
    with pytest.raises(ValueError, match="no notebook-types baseline"):
        enabled_rules(repo, absent)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_the_run_needs_a_positive_finite_deadline(timeout: float) -> None:
    """Refuse a deadline that cannot bound the run."""
    with pytest.raises(ValueError, match="deadline must be positive and finite"):
        notebook_canary_findings(ROOT, sys.executable, "0" * 40, PATHS, timeout=timeout)


@pytest.mark.parametrize(
    "paths",
    [
        PATHS[1:],
        PATHS[:1],
        (*PATHS, f"{FIXTURE}/notes.txt"),
        (*PATHS, f"{FIXTURE}/nested/canary.ipynb.json"),
        (PATHS[0], f"{FIXTURE}/refused-escape.ipynb"),
    ],
)
def test_the_fixture_files_must_be_the_canary_and_refused_notebooks(
    paths: tuple[str, ...],
) -> None:
    """Refuse a fixture without exactly one canary and some refused notebooks."""
    with pytest.raises(ValueError, match="Invalid notebook canary fixture files"):
        notebook_canary_findings(ROOT, sys.executable, "0" * 40, paths, timeout=60)


def test_an_exhausted_deadline_is_refused() -> None:
    """Refuse a run whose deadline passes before the analysis starts."""
    head = git(ROOT, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="exhausted its whole deadline"):
        notebook_canary_findings(ROOT, sys.executable, head, PATHS, timeout=1e-9)


def test_the_lane_interpreter_cannot_run_the_canaries() -> None:
    """Refuse by name a canary run by the lane's own setup-python build."""
    head = git(ROOT, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="The running Python binary differs"):
        notebook_canary_findings(ROOT, sys.executable, head, PATHS, timeout=300)


def test_only_this_paths_escapes_and_awaits_become_keys() -> None:
    """Key the escape and await advisories of the path; skip everything else."""
    path = f"{FIXTURE}/canary.ipynb.json"
    assert _advisory_keys(
        path,
        (
            f"notebook-shell-escape:{path}:2:1",
            f"notebook-native-top-level-await:{path}:15:2",
            "notebook-shell-escape:notebooks/other.ipynb:1:1",
            f"notebook-projection-sha256:{path}:{'0' * 64}",
            "a note without a colon",
        ),
    ) == [
        FindingKey(path, "notebook-shell-escape", 1, None, 2),
        FindingKey(path, "notebook-top-level-await", 2, None, 15),
    ]


def test_the_refused_statement_is_the_last_code_line(tmp_path: Path) -> None:
    """Locate the last non-empty code cell's last line; refuse a fixture without one."""
    cases = {
        "listed.json": notebook("value = 1\n", ["first\n", "!ls\n\n"]),
        "empty.json": {**notebook(), "cells": [{"cell_type": "code", "source": " "}]},
        "cellless.json": {"nbformat": 4},
    }
    for name, value in cases.items():
        (tmp_path / name).write_text(json.dumps(value))
    assert _last_statement(tmp_path, "listed.json") == (3, 2)
    with pytest.raises(ValueError, match="has no code cell"):
        _last_statement(tmp_path, "empty.json")
    with pytest.raises(ValueError, match="has no cells"):
        _last_statement(tmp_path, "cellless.json")


RUN = """
import json, sys
from argparse import Namespace
from pathlib import Path
from tools.quality_ratchet_canaries import run_profile_canaries
run = run_profile_canaries(
    Path.cwd(),
    Namespace(python=sys.executable, base_ref="HEAD", timeout=600.0),
    "notebook-types",
)
print(json.dumps({
    "rules": run.enabled_rules,
    "fixture": run.fixture_digest,
    "platform": run.platform,
    "engine": run.engine_digest,
    "keys": [
        [Path(k.path).name, k.rule, k.cell, k.line, k.column] for k in run.findings
    ],
}))
"""

SEEDED = [
    ["canary.ipynb.json", "assignment", 2, 2, 31],
    ["canary.ipynb.json", "assignment", 16, 3, 22],
    ["canary.ipynb.json", "attr-defined", 5, 1, 21],
    ["canary.ipynb.json", "call-arg", 10, 5, 12],
    ["canary.ipynb.json", "call-overload", 13, 2, 11],
    ["canary.ipynb.json", "import-not-found", 8, 1, 1],
    ["canary.ipynb.json", "import-untyped", 9, 1, 1],
    ["canary.ipynb.json", "no-any-return", 11, 2, 5],
    ["canary.ipynb.json", "no-untyped-call", 4, 2, 5],
    ["canary.ipynb.json", "no-untyped-def", 3, 1, 1],
    ["canary.ipynb.json", "notebook-shell-escape", 2, 1, None],
    ["canary.ipynb.json", "notebook-top-level-await", 15, 2, None],
    ["canary.ipynb.json", "operator", 7, 1, 17],
    ["canary.ipynb.json", "type-arg", 6, 1, 27],
    ["canary.ipynb.json", "union-attr", 12, 2, 12],
    ["canary.ipynb.json", "var-annotated", 14, 1, 1],
    [
        "refused-assigned-escape.ipynb.json",
        "notebook-refused-python-syntax",
        1,
        2,
        None,
    ],
    ["refused-double-escape.ipynb.json", "notebook-refused-shell-escape", 1, 2, None],
    ["refused-escape.ipynb.json", "notebook-refused-shell-escape", 1, 2, None],
]


def test_the_registered_runner_fires_every_rule_with_the_pinned_engine() -> None:
    """Fire every derived code and projection item at its cell and line."""
    run = run_quality(RUN)
    assert isinstance(run, dict)
    head = git(ROOT, "rev-parse", "HEAD")
    assert run["rules"] == list(enabled_rules(ROOT, head))
    assert {key[1] for key in run["keys"]} == set(run["rules"])
    assert run["keys"] == SEEDED
    assert run["fixture"] == fixture_digest(ROOT, "notebook-types")
    assert run["platform"] == "linux-x64"
    assert run["engine"] == PYTHON_ARTEFACTS["linux-x64"]["binary"]


REFUSE = """
import json, sys
from pathlib import Path
from tools.quality_ratchet_notebook_canary import _refused_key
try:
    key = _refused_key(Path(sys.argv[1]), sys.executable, sys.argv[2], 300.0)
except ValueError as error:
    print(json.dumps(str(error)))
else:
    print(json.dumps([key.rule, key.cell, key.line]))
"""


@pytest.mark.parametrize(
    ("value", "outcome"),
    [
        (notebook("# comment\n!ls $HOME\n"), ["notebook-refused-shell-escape", 2, 2]),
        (notebook("listing = !ls\n"), ["notebook-refused-python-syntax", 2, 1]),
        (notebook("value: int = 1\n"), "was not refused"),
        (notebook("!ls\n", language="julia"), "refused for another reason"),
        (notebook("!ls $HOME\nvalue = 1\n"), "refused at other coordinates"),
    ],
)
def test_a_refused_fixture_is_keyed_by_its_reason_and_position(
    typing_repository: Path, value: dict[str, object], outcome: object
) -> None:
    """Key a refusal; refuse acceptance, another reason or other coordinates."""
    path = "tests/fixtures/refused-case.ipynb.json"
    (typing_repository / path).parent.mkdir(parents=True)
    (typing_repository / path).write_text(json.dumps(value))
    commit_typing_sources(typing_repository)
    result = run_quality(REFUSE, str(typing_repository), path)
    if isinstance(outcome, list):
        assert result == outcome
    else:
        assert isinstance(result, str) and str(outcome) in result, result
