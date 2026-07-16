# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI lazy pro-tier dispatch tests (ladder P2)

from __future__ import annotations

import pytest


def test_lazy_handler_resolves_and_runs_paid_subcommand() -> None:
    import director_ai.cli as cli

    handler = cli._lazy_handler("._cli_bench", "_cmd_validate_data")
    # resolves the real handler and propagates its usage exit
    with pytest.raises(SystemExit):
        handler([])


def test_lazy_handler_missing_module_exits_with_tier_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import director_ai.cli as cli

    def _boom(name: str, package: str | None = None) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(cli, "import_module", _boom)
    handler = cli._lazy_handler("._cli_serve", "_cmd_serve")
    with pytest.raises(SystemExit, match="requires the advanced tier"):
        handler([])


def test_module_getattr_resolves_ingest_limit_lazily() -> None:
    import director_ai.cli as cli

    assert cli._INGEST_MAX_FILE_SIZE > 0


def test_module_getattr_unknown_name_raises_attribute_error() -> None:
    import director_ai.cli as cli

    with pytest.raises(AttributeError, match="has no attribute 'NoSuchThing'"):
        _ = cli.NoSuchThing


def test_paid_specs_use_lazy_dispatch_not_module_imports() -> None:
    """The free CLI module must not import any pro-tier ``_cli_*`` module."""
    import sys
    from importlib import import_module

    for mod in list(sys.modules):
        if mod.startswith("director_ai._cli_") or mod == "director_ai.cli":
            del sys.modules[mod]
    import_module("director_ai.cli")  # fresh import — no paid modules may load

    paid = [m for m in sys.modules if m.startswith("director_ai._cli_")]
    assert paid == ["director_ai._cli_verify"], paid
