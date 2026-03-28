# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — embeds
# (C) 1998-2026 Miroslav Sotek. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import discord

BLURPLE = 0x5865F2
GREEN = 0x2ECC71
RED = 0xE74C3C
DOCKER_BLUE = 0x3498DB
AMBER = 0xF1C40F

REPO = "anulum/director-ai"
DOCS_URL = "https://anulum.github.io/director-ai"
PYPI_URL = "https://pypi.org/project/director-ai"
GHCR_URL = "https://ghcr.io/anulum/director-ai"


def release_embed(
    version: str,
    changelog_bullets: list[str],
    release_url: str,
) -> discord.Embed:
    desc = "\n".join(f"- {b}" for b in changelog_bullets[:5]) or "See release notes."
    em = discord.Embed(
        title=f"Director-AI v{version} released",
        description=desc,
        color=BLURPLE,
        url=release_url,
        timestamp=datetime.now(UTC),
    )
    em.add_field(
        name="Install",
        value=f"```\npip install director-ai=={version}\n```",
        inline=False,
    )
    em.add_field(
        name="Links",
        value=(
            f"[PyPI]({PYPI_URL}/{version}/) · "
            f"[Release]({release_url}) · "
            f"[Docs]({DOCS_URL}) · "
            f"[Docker]({GHCR_URL})"
        ),
        inline=False,
    )
    em.set_footer(text=REPO, icon_url="https://github.com/anulum.png")
    return em


def ci_embed(
    passed: bool,
    branch: str,
    sha: str,
    run_url: str,
    job_summary: str = "",
) -> discord.Embed:
    status = "passed" if passed else "FAILED"
    color = GREEN if passed else RED
    em = discord.Embed(
        title=f"CI {status}",
        color=color,
        url=run_url,
        timestamp=datetime.now(UTC),
    )
    em.add_field(name="Branch", value=f"`{branch}`", inline=True)
    em.add_field(name="Commit", value=f"`{sha[:7]}`", inline=True)
    em.add_field(name="Run", value=f"[View]({run_url})", inline=True)
    if job_summary:
        em.add_field(name="Jobs", value=job_summary, inline=False)
    em.set_footer(text=REPO, icon_url="https://github.com/anulum.png")
    return em


def status_embed(
    pypi_version: str,
    ci_conclusion: str | None,
    ci_run_url: str | None,
) -> discord.Embed:
    ci_icon = {
        "success": "\u2705",
        "failure": "\u274c",
        "cancelled": "\u23f9\ufe0f",
    }
    em = discord.Embed(
        title="Director-AI Status",
        color=BLURPLE,
        timestamp=datetime.now(UTC),
    )
    em.add_field(name="PyPI", value=f"`{pypi_version}`", inline=True)
    if ci_conclusion and ci_run_url:
        icon = ci_icon.get(ci_conclusion, "\u2753")
        em.add_field(
            name="CI",
            value=f"{icon} {ci_conclusion} — [run]({ci_run_url})",
            inline=True,
        )
    em.add_field(
        name="Links",
        value=(
            f"[Docs]({DOCS_URL}) · "
            f"[PyPI]({PYPI_URL}) · "
            f"[GitHub](https://github.com/{REPO})"
        ),
        inline=False,
    )
    em.set_footer(text=REPO, icon_url="https://github.com/anulum.png")
    return em


def docker_embed(
    cpu_tags: list[str],
    gpu_tags: list[str],
) -> discord.Embed:
    em = discord.Embed(
        title="Docker images published",
        color=DOCKER_BLUE,
        timestamp=datetime.now(UTC),
    )
    if cpu_tags:
        em.add_field(
            name="CPU",
            value="\n".join(f"`{t}`" for t in cpu_tags),
            inline=True,
        )
    if gpu_tags:
        em.add_field(
            name="GPU",
            value="\n".join(f"`{t}`" for t in gpu_tags),
            inline=True,
        )
    em.add_field(
        name="Pull",
        value=f"```\ndocker pull {GHCR_URL}:latest\n```",
        inline=False,
    )
    em.set_footer(text=REPO, icon_url="https://github.com/anulum.png")
    return em


def welcome_embed(display_name: str) -> discord.Embed:
    em = discord.Embed(
        title=f"Welcome, {display_name}!",
        description=(
            "**Director-AI** is a real-time LLM hallucination guardrail "
            "with NLI + RAG fact-checking and token-level streaming halt.\n\n"
            "Get started:\n"
            "- `/quickstart` — 6-line guard snippet\n"
            "- `/install` — installation options\n"
            "- `/docs` — full documentation\n"
            "- `/profiles` — domain presets (medical, finance, legal, ...)\n\n"
            f"[GitHub](https://github.com/{REPO}) · "
            f"[Docs]({DOCS_URL}) · "
            f"[PyPI]({PYPI_URL})"
        ),
        color=BLURPLE,
        timestamp=datetime.now(UTC),
    )
    em.set_footer(text=REPO, icon_url="https://github.com/anulum.png")
    return em
