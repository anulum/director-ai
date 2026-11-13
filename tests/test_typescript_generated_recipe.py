# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real native SDK recipe and process lifecycle tests
"""Exercise actual SDK producer inputs, native commands and finite-process refusal."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.typescript_generated_recipe import (
    _signal_group,
    capture_toolchain,
    run_native,
)

ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / "packages/vercel-ai"


@dataclass(frozen=True)
class SDKCase:
    """Identify a real SDK source/Git fixture and its actual generated artefacts.

    Attributes
    ----------
    root : Path
        Actual committed SDK source fixture, with production source/config bytes.
    sdk : Path
        SDK package and separate native documentation output root.
    archive : Path
        Actual npm pack archive produced with original prepack/build hooks.
    """

    root: Path
    sdk: Path
    archive: Path


@pytest.fixture(scope="module")
def sdk_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[SDKCase, None, None]:
    """Build the real unchanged SDK under its genuine installed ancestor graph.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Runner-owned location for original command receipts and package output.

    Yields
    ------
    SDKCase
        Real Git source fixture with actual strict compiler, TypeDoc and npm output.
    """
    default = SDK / "build"
    default.mkdir(exist_ok=True)
    parent = Path(os.environ.get("DIRECTOR_GENERATED_TEST_ALLOCATION", str(default)))
    traces = tmp_path_factory.mktemp("sdk-native-receipts")
    environment = os.environ.copy()
    environment["PATH"] = (
        str(SDK / "node_modules/.bin") + os.pathsep + environment["PATH"]
    )
    with tempfile.TemporaryDirectory(prefix="sdk-native-", dir=parent) as temporary:
        root = Path(temporary)
        target = root / "packages/vercel-ai"
        target.mkdir(parents=True)
        paths = (
            subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                    "--",
                    "packages/vercel-ai",
                ],
                cwd=ROOT,
            )
            .decode()
            .split("\0")
        )
        for relative in paths:
            if relative:
                source = ROOT / relative
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        shutil.copy2(ROOT / ".gitignore", root / ".gitignore")
        commands = [
            ["git", "init", "--initial-branch=main"],
            ["git", "config", "user.name", "Native fixture"],
            ["git", "config", "user.email", "fixture@example.test"],
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://github.com/anulum/director-ai.git",
            ],
            ["git", "add", ".gitignore", "packages/vercel-ai"],
            ["git", "commit", "-m", "original SDK fixture"],
        ]
        for index, command in enumerate(commands):
            run_native(
                command,
                root,
                time.monotonic() + 30,
                environment,
                traces / f"git-{index}.json",
            )
        generators = [
            [
                "node",
                str(SDK / "node_modules/typescript/bin/tsc"),
                "--project",
                str(target / "tsconfig.json"),
            ],
            [
                "node",
                str(SDK / "node_modules/typedoc/bin/typedoc"),
                "--options",
                str(target / "typedoc.json"),
            ],
            ["npm", "pack", "--offline", "--json", "--pack-destination", str(traces)],
        ]
        for index, command in enumerate(generators):
            run_native(
                command,
                target,
                time.monotonic() + 90,
                environment,
                traces / f"producer-{index}.json",
            )
        packages = list(traces.glob("*.tgz"))
        assert len(packages) == 1
        yield SDKCase(root, target, packages[0])


@pytest.fixture
def sdk_copy(sdk_case: SDKCase) -> Generator[SDKCase, None, None]:
    """Fork only the actual SDK/Git fixture for controlled real input mutations.

    Parameters
    ----------
    sdk_case : SDKCase
        Real native source and production-generated output oracle.

    Yields
    ------
    SDKCase
        Independent mutable source fixture. Supplied documentation keeps its
        original checkout context until a test explicitly regenerates it.
    """
    with tempfile.TemporaryDirectory(
        prefix="sdk-mutation-", dir=sdk_case.root.parent
    ) as temporary:
        root = Path(temporary) / "source"
        shutil.copytree(sdk_case.root, root)
        yield SDKCase(root, root / "packages/vercel-ai", sdk_case.archive)


def test_real_installed_tool_graph_is_complete() -> None:
    """Bind actual Node and thousands of genuine installed dependency bytes."""
    result = capture_toolchain(SDK, "node", deadline=time.monotonic() + 30)
    assert result.executable.is_file()
    assert result.executable_mode & 0o111
    assert "typescript/lib/typescript.js" in result.dependencies
    assert "typedoc/package.json" in result.dependencies
    assert len(result.executable_sha256) == 64


def test_missing_and_empty_dependency_installations_refuse(tmp_path: Path) -> None:
    """Reject absent or empty installed inputs before any native compatibility claim.

    Parameters
    ----------
    tmp_path : Path
        Actual runner-owned filesystem for incomplete-installation inputs.
    """
    with pytest.raises(FileNotFoundError):
        capture_toolchain(tmp_path, "node", deadline=time.monotonic() + 5)
    (tmp_path / "node_modules").mkdir()
    with pytest.raises(ValueError, match="empty"):
        capture_toolchain(tmp_path, "node", deadline=time.monotonic() + 5)


@pytest.mark.parametrize(
    "kind", ["outside-link", "internal-link", "fifo", "missing-node", "directory-alias"]
)
def test_actual_dependency_entry_contracts(tmp_path: Path, kind: str) -> None:
    """Qualify native filesystem link bytes and refuse invalid real installations.

    Parameters
    ----------
    tmp_path : Path
        Real runner-owned installation fixture for negative boundary states.
    kind : str
        Actual filesystem or native executable fault to exercise.
    """
    modules = tmp_path / "node_modules"
    modules.mkdir()
    (modules / "present").write_text("original dependency bytes")
    if kind == "outside-link":
        (modules / "escape").symlink_to(tmp_path)
        with pytest.raises(ValueError, match="leaves"):
            capture_toolchain(tmp_path, "node", deadline=time.monotonic() + 5)
    elif kind == "internal-link":
        (modules / "alias").symlink_to("present")
        result = capture_toolchain(tmp_path, "node", deadline=time.monotonic() + 5)
        assert result.dependencies["alias"] != result.dependencies["present"]
    elif kind == "fifo":
        os.mkfifo(modules / "pipe")
        with pytest.raises(ValueError, match="nonregular"):
            capture_toolchain(tmp_path, "node", deadline=time.monotonic() + 5)
    elif kind == "missing-node":
        with pytest.raises(ValueError, match="absent"):
            capture_toolchain(
                tmp_path, "/absent/native-node", deadline=time.monotonic() + 5
            )
    else:
        alias = tmp_path / "install-alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="exact real"):
            capture_toolchain(alias, "node", deadline=time.monotonic() + 5)


def test_real_native_command_preserves_output_and_finite_failure(
    tmp_path: Path,
) -> None:
    """Run actual Node and retain its real stdout, nonzero result and receipt.

    Parameters
    ----------
    tmp_path : Path
        Actual source-free native receipt directory owned by the runner.
    """
    result = run_native(
        ["node", "--version"],
        ROOT,
        time.monotonic() + 5,
        os.environ.copy(),
        tmp_path / "node.json",
    )
    assert result.strip() == "v26.13.0"
    with pytest.raises(ValueError, match="did not qualify"):
        run_native(
            ["node", "--unknown-native-option"],
            ROOT,
            time.monotonic() + 5,
            os.environ.copy(),
            tmp_path / "refusal.json",
        )
    receipt = json.loads((tmp_path / "refusal.json").read_text())
    assert receipt["exit"] != 0 and receipt["stderr"]
    with pytest.raises(ValueError, match="deadline"):
        run_native(
            ["node", "--version"],
            ROOT,
            time.monotonic() - 1,
            os.environ.copy(),
            tmp_path / "expired.json",
        )
    assert not (tmp_path / "expired.json").exists()


@pytest.mark.parametrize("ignore_term", [False, True])
def test_actual_owned_process_deadline_reaps_group(
    tmp_path: Path, ignore_term: bool
) -> None:
    """Terminate an actual sleeping child, escalating only when it ignores TERM.

    Parameters
    ----------
    tmp_path : Path
        Owned process receipt directory.
    ignore_term : bool
        Whether the actual child deliberately ignores its graceful stop signal.
    """
    code = (
        "import signal,time;"
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_term else "")
        + "print('native child ready',flush=True);time.sleep(30)"
    )
    with pytest.raises(ValueError, match="did not qualify"):
        run_native(
            [sys.executable, "-c", code],
            ROOT,
            time.monotonic() + 0.7,
            os.environ.copy(),
            tmp_path / "timeout.json",
        )
    receipt = json.loads((tmp_path / "timeout.json").read_text())
    assert receipt["whole_deadline_expired"]
    assert "native child ready" in receipt["stdout"]
    with pytest.raises(ProcessLookupError):
        os.kill(receipt["pid"], 0)


@pytest.mark.parametrize("ignore_term", [False, True])
def test_actual_interrupt_reaps_the_owned_native_group(
    tmp_path: Path, ignore_term: bool
) -> None:
    """Deliver real SIGINT to an isolated caller and retain its reaped-child receipt.

    Parameters
    ----------
    tmp_path : Path
        Actual original readiness marker, command receipt and caller program.
    ignore_term : bool
        Whether the real native child requires escalation to SIGKILL.
    """
    marker = tmp_path / "child-ready"
    receipt = tmp_path / "interrupted.json"
    native = (
        "import os,signal,time;from pathlib import Path;"
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_term else "")
        + f"print('native child ready',flush=True);Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(10)"
    )
    caller = f"""
import os,signal,sys,threading,time
from pathlib import Path
from tools.typescript_generated_recipe import run_native
marker = Path({str(marker)!r})
def interrupt_when_ready():
    while not marker.exists():
        time.sleep(0.01)
    os.kill(os.getpid(), signal.SIGINT)
threading.Thread(target=interrupt_when_ready, daemon=True).start()
try:
    run_native([sys.executable, '-c', {native!r}], Path({str(ROOT)!r}), time.monotonic() + 30, os.environ.copy(), Path({str(receipt)!r}))
except KeyboardInterrupt:
    raise SystemExit(130)
"""
    result = subprocess.run(
        [sys.executable, "-c", caller],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 130, result.stdout + result.stderr
    original = json.loads(receipt.read_text())
    assert original["interrupted"] is True
    assert original["whole_deadline_expired"] is False
    assert original["exit"] < 0
    assert "native child ready" in original["stdout"]
    assert marker.read_text() == str(original["pid"])
    with pytest.raises(ProcessLookupError):
        os.kill(original["pid"], 0)


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("emit_output", [False, True])
def test_escaped_output_holder_cannot_extend_cleanup_deadline(
    tmp_path: Path, interrupt: bool, emit_output: bool
) -> None:
    """Bound a real escaped pipe holder and retain interruption after leader exit.

    Parameters
    ----------
    tmp_path : Path
        Owned marker, output receipt and isolated caller result directory.
    interrupt : bool
        Send actual SIGINT after the native leader exits instead of expiring time.
    emit_output : bool
        Capture real partial stdout/stderr or exercise genuinely empty streams.
    """
    marker = tmp_path / "escaped-child"
    receipt = tmp_path / "escaped.json"
    outcome = tmp_path / "outcome.json"
    native = f"""
import os,sys,time
from pathlib import Path
marker = Path({str(marker)!r})
if os.fork() == 0:
    os.setsid()
    if {emit_output!r}:
        print('escaped stdout', flush=True)
        print('escaped stderr', file=sys.stderr, flush=True)
    marker.write_text(str(os.getpid()))
    time.sleep(12)
    os._exit(0)
for attempt in range(300):
    if marker.exists():
        os._exit(0)
    time.sleep(0.01)
os._exit(1)
"""
    caller = f"""
import ctypes,json,os,signal,sys,threading,time
from pathlib import Path
from tools.typescript_generated_recipe import run_native
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
marker = Path({str(marker)!r})
if {interrupt!r}:
    def interrupt_when_ready():
        for attempt in range(300):
            if marker.exists():
                time.sleep(0.1)
                os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(0.01)
    threading.Thread(target=interrupt_when_ready, daemon=True).start()
start = time.monotonic()
try:
    run_native([sys.executable, '-c', {native!r}], Path({str(ROOT)!r}), start + 0.6, os.environ.copy(), Path({str(receipt)!r}))
except (ValueError, KeyboardInterrupt) as error:
    result = {{'error': type(error).__name__, 'elapsed': time.monotonic() - start}}
finally:
    child = int(marker.read_text())
    os.kill(child, signal.SIGKILL)
    waited, status = os.waitpid(child, 0)
    assert waited == child and os.waitstatus_to_exitcode(status) == -signal.SIGKILL
Path({str(outcome)!r}).write_text(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", caller],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=18,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(outcome.read_text())
    assert result["error"] == ("KeyboardInterrupt" if interrupt else "ValueError")
    assert result["elapsed"] < 7.6
    original = json.loads(receipt.read_text())
    assert original["exit"] == 0
    assert original["interrupted"] is interrupt
    assert original["whole_deadline_expired"] is (not interrupt)
    assert original["output_complete"] is False
    assert original["reaped"] is True
    assert original["stdout"] == ("escaped stdout\n" if emit_output else "")
    assert original["stderr"] == ("escaped stderr\n" if emit_output else "")
    for pid in [original["pid"], int(marker.read_text())]:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_signal_group_reports_a_group_that_already_ended() -> None:
    """A group that exited before the deadline signal is reported, not hidden.

    The command runs in its own session, exits and is reaped; signalling its
    former group then meets ``ProcessLookupError``, which the helper turns
    into ``False``. A live group receives the signal and reports ``True``.
    """
    import signal

    ended = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    ended.wait(timeout=30)
    assert _signal_group(ended.pid, signal.SIGTERM) is False

    live = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        assert _signal_group(live.pid, signal.SIGTERM) is True
        assert live.wait(timeout=30) == -signal.SIGTERM
    finally:
        if live.poll() is None:
            live.kill()
            live.wait(timeout=30)
