# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for gem CLI commands pipeline."""

from __future__ import annotations

import sys

import pytest


class TestVerifyNumericCli:
    def test_clean_text(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["director-ai", "verify-numeric", "The sky is blue."]
        )
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "Valid:" in out
        assert "True" in out

    def test_bad_arithmetic(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["director-ai", "verify-numeric", "Revenue grew 50% from $100 to $120."],
        )
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "False" in out
        assert "arithmetic" in out

    def test_no_args_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["director-ai", "verify-numeric"])
        from director_ai.cli import main as cli_main

        with pytest.raises(SystemExit, match="1"):
            cli_main()


class TestVerifyReasoningCli:
    def test_valid_chain(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "director-ai",
                "verify-reasoning",
                "Step 1: All mammals are warm-blooded.",
                "Step 2: Dogs are mammals.",
                "Step 3: Therefore, dogs are warm-blooded.",
            ],
        )
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "Chain valid:" in out
        assert "Steps:" in out

    def test_no_args_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["director-ai", "verify-reasoning"])
        from director_ai.cli import main as cli_main

        with pytest.raises(SystemExit, match="1"):
            cli_main()


class TestTemporalFreshnessCli:
    def test_position_claim(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["director-ai", "temporal-freshness", "The CEO of Apple is Tim Cook."],
        )
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "Has temporal claims:" in out
        assert "True" in out

    def test_no_args_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["director-ai", "temporal-freshness"])
        from director_ai.cli import main as cli_main

        with pytest.raises(SystemExit, match="1"):
            cli_main()


class TestCheckStepCli:
    def test_normal_step(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "director-ai",
                "check-step",
                "search revenue data",
                "search",
                "revenue data",
            ],
        )
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "Step:" in out
        assert "Halt:" in out

    def test_no_args_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["director-ai", "check-step"])
        from director_ai.cli import main as cli_main

        with pytest.raises(SystemExit, match="1"):
            cli_main()

    def test_one_arg_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["director-ai", "check-step", "goal_only"])
        from director_ai.cli import main as cli_main

        with pytest.raises(SystemExit, match="1"):
            cli_main()


class TestConsensusCli:
    def test_agreement(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "director-ai",
                "consensus",
                "gpt:Paris is the capital",
                "claude:Paris is the capital",
            ],
        )
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "Agreement:" in out
        assert "Consensus:" in out
        assert "True" in out

    def test_disagreement(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "director-ai",
                "consensus",
                "a:Apples grow on trees",
                "b:The speed of light is fast",
            ],
        )
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "DISAGREE" in out

    def test_no_args_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["director-ai", "consensus"])
        from director_ai.cli import main as cli_main

        with pytest.raises(SystemExit, match="1"):
            cli_main()

    def test_one_arg_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["director-ai", "consensus", "gpt:only one"])
        from director_ai.cli import main as cli_main

        with pytest.raises(SystemExit, match="1"):
            cli_main()

    def test_bad_format_exits(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["director-ai", "consensus", "no_colon", "also_no_colon"],
        )
        from director_ai.cli import main as cli_main

        with pytest.raises(SystemExit, match="1"):
            cli_main()


class TestAdversarialTestCli:
    def test_runs(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["director-ai", "adversarial-test", "Tell me about this topic"],
        )
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "Patterns:" in out
        assert "Detected:" in out
        assert "Rate:" in out

    def test_default_prompt(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["director-ai", "adversarial-test"])
        from director_ai.cli import main as cli_main

        cli_main()
        out = capsys.readouterr().out
        assert "Patterns:" in out
