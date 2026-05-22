# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — full benchmark campaign tests

from __future__ import annotations

from benchmarks import full_benchmark_campaign as campaign


def test_render_markdown_has_table_and_summary() -> None:
    payload = {
        "generated_utc": "2026-05-22T00:00:00+00:00",
        "git_commit": "abc",
        "python_version": "3.12.3",
        "platform": "linux",
        "cases": [
            {
                "name": "case_a",
                "category": "cat",
                "status": "passed",
                "duration_s": 1.23,
                "return_code": 0,
                "artifact_files": ["benchmarks/results/x.json"],
            }
        ],
    }
    md = campaign._render_markdown(payload)
    assert "Full Benchmark Campaign" in md
    assert "`case_a`" in md
    assert "Passed: **1/1**" in md


def test_run_campaign_strict_exit(monkeypatch) -> None:
    fake = campaign.CaseResult(
        name="x",
        category="y",
        status="failed",
        return_code=1,
        duration_s=0.1,
        command=["python"],
        artifact_files=[],
        stdout_tail="",
        stderr_tail="",
    )
    monkeypatch.setattr(campaign, "_campaign_cases", lambda: [campaign.CampaignCase("x", ["python"], 1, "y")])
    monkeypatch.setattr(campaign, "_run_case", lambda _case: fake)
    payload, code = campaign.run_campaign(strict=True)
    assert payload["benchmark"] == "full_benchmark_campaign"
    assert code == 2
