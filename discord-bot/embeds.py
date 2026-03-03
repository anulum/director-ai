# (C) 1998-2026 Miroslav Sotek. All rights reserved.

from __future__ import annotations

import discord

BLURPLE = 0x5865F2
GREEN = 0x2ECC71
RED = 0xE74C3C
DOCKER_BLUE = 0x3498DB


def release_embed(
    version: str,
    changelog_bullets: list[str],
    release_url: str,
    repo: str = "anulum/director-ai",
) -> discord.Embed:
    desc = "\n".join(f"- {b}" for b in changelog_bullets[:3]) or "See release notes."
    em = discord.Embed(
        title=f"Director-AI v{version} released",
        description=desc,
        color=BLURPLE,
        url=release_url,
    )
    em.add_field(name="Install", value=f"```\npip install director-ai=={version}\n```", inline=False)
    em.add_field(
        name="Links",
        value=(
            f"[PyPI](https://pypi.org/project/director-ai/{version}/) · "
            f"[Release]({release_url}) · "
            f"[Docs](https://anulum.github.io/director-ai/)"
        ),
        inline=False,
    )
    em.set_footer(text=repo)
    return em


def ci_embed(
    passed: bool,
    branch: str,
    sha: str,
    run_url: str,
    repo: str = "anulum/director-ai",
) -> discord.Embed:
    status = "passed" if passed else "failed"
    em = discord.Embed(
        title=f"CI {status}",
        color=GREEN if passed else RED,
        url=run_url,
    )
    em.add_field(name="Branch", value=f"`{branch}`", inline=True)
    em.add_field(name="Commit", value=f"`{sha[:7]}`", inline=True)
    em.add_field(name="Run", value=f"[View]({run_url})", inline=True)
    em.set_footer(text=repo)
    return em


def status_embed(
    pypi_version: str,
    ci_conclusion: str | None,
    ci_run_url: str | None,
) -> discord.Embed:
    em = discord.Embed(title="Director-AI Status", color=BLURPLE)
    em.add_field(name="PyPI", value=f"`{pypi_version}`", inline=True)
    if ci_conclusion and ci_run_url:
        em.add_field(name="CI", value=f"{ci_conclusion} — [run]({ci_run_url})", inline=True)
    em.add_field(
        name="Docs",
        value="[anulum.github.io/director-ai](https://anulum.github.io/director-ai/)",
        inline=False,
    )
    return em


def docker_embed(
    cpu_tags: list[str],
    gpu_tags: list[str],
    repo: str = "anulum/director-ai",
) -> discord.Embed:
    em = discord.Embed(title="Docker images published", color=DOCKER_BLUE)
    if cpu_tags:
        em.add_field(name="CPU", value="\n".join(f"`{t}`" for t in cpu_tags), inline=True)
    if gpu_tags:
        em.add_field(name="GPU", value="\n".join(f"`{t}`" for t in gpu_tags), inline=True)
    em.set_footer(text=repo)
    return em
