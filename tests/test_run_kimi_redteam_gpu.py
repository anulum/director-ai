# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Teardown-safety tests for the JarvisLabs KIMI red-team GPU runner.

The runner's one cost-critical job is that a finished or failed run never leaves
a paid GPU running. These tests exercise ``destroy_instance``'s retry +
verify-at-source contract with plain protocol-preserving fakes (no mocks), so
the exact leak that once happened — ``destroy()`` raising a transient network
error while the instance kept running — is provably caught.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

import tools.run_kimi_redteam_gpu as runner


class _Listed:
    """Minimal stand-in for a JarvisLabs instance in a ``get_instances`` list."""

    def __init__(self, machine_id: str) -> None:
        self.machine_id = machine_id


def _lister(live: set[str]) -> Callable[[], list[_Listed]]:
    """Return a ``list_instances`` callable reporting whatever is in *live*."""

    def _list() -> list[_Listed]:
        return [_Listed(mid) for mid in sorted(live)]

    return _list


class _FakeInstance:
    """A JarvisLabs instance whose ``destroy()`` behaviour is scriptable.

    *live* is the shared at-source registry the lister reads. ``destroy()``
    removes ``machine_id`` from it once ``destroy_calls`` reaches *removes_after*
    (modelling the server-side teardown) and raises *raise_times* times first
    (modelling a transient client-side network error). The removal happens even
    when the call raises, reproducing the real leak shape: the API tore the GPU
    down but the client saw only ``RemoteDisconnected``.
    """

    def __init__(
        self,
        machine_id: str,
        *,
        live: set[str],
        raise_times: int = 0,
        removes_after: int = 1,
    ) -> None:
        self.machine_id = machine_id
        self._live = live
        self._live.add(machine_id)
        self._raise_times = raise_times
        self._removes_after = removes_after
        self.destroy_calls = 0

    def destroy(self) -> str:
        self.destroy_calls += 1
        if self.destroy_calls >= self._removes_after:
            self._live.discard(self.machine_id)
        if self.destroy_calls <= self._raise_times:
            raise ConnectionError("RemoteDisconnected: Remote end closed connection")
        return "destroyed"


def _no_sleep(_seconds: float) -> None:
    """Backoff stub so the retry loop never actually waits."""


def test_live_machine_ids_reads_ids_at_source() -> None:
    live = {"111", "222"}
    assert runner._live_machine_ids(_lister(live)) == {"111", "222"}


def test_destroy_confirmed_on_first_attempt() -> None:
    live: set[str] = set()
    inst = _FakeInstance("777", live=live)

    ok = runner.destroy_instance(inst, list_instances=_lister(live), sleep=_no_sleep)

    assert ok is True
    assert inst.destroy_calls == 1
    assert "777" not in live


def test_transient_error_but_instance_actually_gone_is_success() -> None:
    """The exact leak scenario: destroy() raises yet the GPU really tore down."""
    live: set[str] = set()
    inst = _FakeInstance("449486", live=live, raise_times=1, removes_after=1)

    ok = runner.destroy_instance(inst, list_instances=_lister(live), sleep=_no_sleep)

    # destroy() raised, but the at-source re-check proves it is gone -> success.
    assert ok is True
    assert inst.destroy_calls == 1
    assert "449486" not in live


def test_silent_success_that_did_not_tear_down_is_retried() -> None:
    """destroy() returns normally but the GPU is still live -> retry until gone."""
    live: set[str] = set()
    inst = _FakeInstance("333", live=live, raise_times=0, removes_after=2)

    ok = runner.destroy_instance(inst, list_instances=_lister(live), sleep=_no_sleep)

    assert ok is True
    assert inst.destroy_calls == 2
    assert "333" not in live


def test_persistent_leak_returns_false_and_logs_critical(
    caplog: pytest.LogCaptureFixture,
) -> None:
    live: set[str] = set()
    inst = _FakeInstance("leaky", live=live, removes_after=99)

    with caplog.at_level("CRITICAL"):
        ok = runner.destroy_instance(
            inst, list_instances=_lister(live), retries=3, sleep=_no_sleep
        )

    assert ok is False
    assert inst.destroy_calls == 3
    assert "leaky" in live
    assert any("LEAKED GPU" in rec.message for rec in caplog.records)


def test_unverifiable_liveness_check_fails_closed() -> None:
    """If the at-source check itself errors, assume leaked (never false-clear)."""
    live: set[str] = set()
    inst = _FakeInstance("904", live=live, removes_after=1)

    def _broken_lister() -> list[_Listed]:
        raise ConnectionError("cannot reach JarvisLabs to verify")

    ok = runner.destroy_instance(
        inst, list_instances=_broken_lister, retries=2, sleep=_no_sleep
    )

    assert ok is False
    assert inst.destroy_calls == 2


def _wire_main(monkeypatch: pytest.MonkeyPatch, *, teardown_ok: bool) -> None:
    """Stub provision/run/download/destroy so main() runs without a real GPU."""

    class _Inst:
        ssh_str = "ssh -p 22 root@host"
        machine_id = "m1"

    monkeypatch.setenv("JARVISLABS_TOKEN", "x" * 43)
    monkeypatch.setattr(runner, "provision", lambda *a, **k: _Inst())
    monkeypatch.setattr(runner, "run_redteam", lambda *a, **k: None)
    monkeypatch.setattr(runner, "download_artefact", lambda *a, **k: None)
    monkeypatch.setattr(runner, "destroy_instance", lambda inst: teardown_ok)


def test_main_returns_zero_on_confirmed_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_main(monkeypatch, teardown_ok=True)
    assert runner.main([]) == 0


def test_main_returns_three_when_teardown_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_main(monkeypatch, teardown_ok=False)
    assert runner.main([]) == 3


def test_main_returns_two_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JARVISLABS_TOKEN", raising=False)
    assert runner.main([]) == 2


def test_ssh_parts_extracts_port_and_host() -> None:
    assert runner._ssh_parts("ssh -p 2222 root@1.2.3.4") == ("2222", "root@1.2.3.4")
    assert runner._ssh_parts("ssh root@1.2.3.4") == ("22", "root@1.2.3.4")


def test_rendered_remote_script_records_provenance() -> None:
    script = runner._render_remote_script("v3.18.1", "benchmarks/foo.py")

    # exact cloned commit is resolved and passed through for artefact provenance
    assert "git rev-parse HEAD" in script
    assert '--git-sha "$GIT_SHA"' in script
    # tag + script placeholders are fully substituted, none left dangling
    assert "--branch v3.18.1 " in script
    assert "benchmarks/foo.py --out" in script
    assert "__TAG__" not in script
    assert "__SCRIPT__" not in script
