# (C) 1998-2026 Miroslav Sotek. All rights reserved.

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

import discord
import httpx
from discord import app_commands
from embeds import status_embed, welcome_embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("director-bot")

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
GUILD_ID = os.environ.get("GUILD_ID", "")

if not BOT_TOKEN or not GUILD_ID:
    sys.exit("BOT_TOKEN and GUILD_ID env vars required")

GUILD = discord.Object(id=int(GUILD_ID))
REPO = "anulum/director-ai"
PYPI_URL = "https://pypi.org/pypi/director-ai/json"
GH_API = f"https://api.github.com/repos/{REPO}"
DOCS_URL = "https://anulum.github.io/director-ai"

# Category → channels mapping
CHANNEL_LAYOUT: dict[str, list[str]] = {
    "Development": ["ci-status", "github-feed"],
    "Community": ["announcements", "general", "support"],
}

PROFILES = {
    "fast": "threshold=0.50, no NLI, 1 candidate — low latency prototyping",
    "thorough": "threshold=0.60, NLI+judge hybrid, 3 candidates — production",
    "research": "threshold=0.70, NLI on, 5 candidates — max accuracy",
    "medical": "threshold=0.75, NLI+reranker, w_fact=0.5 — clinical safety",
    "finance": "threshold=0.70, NLI+reranker, w_fact=0.6 — regulatory",
    "legal": "threshold=0.68, NLI on, w_logic=0.6 — legal reasoning",
    "creative": "threshold=0.40, no NLI, permissive — creative writing",
    "customer_support": "threshold=0.55, no NLI, balanced — helpdesk",
    "summarization": "threshold=0.55, NLI on, balanced — document summaries",
    "lite": "threshold=0.50, heuristic only, 1 candidate — zero deps",
}


class DirectorBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = False
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        register_commands(self.tree)
        self.tree.copy_global_to(guild=GUILD)
        await self.tree.sync(guild=GUILD)


client = DirectorBot()


@client.event
async def on_ready() -> None:
    log.info("Logged in as %s", client.user)
    guild = client.get_guild(int(GUILD_ID))
    if guild is None:
        log.error("Guild %s not found", GUILD_ID)
        return
    await _bootstrap_channels(guild)
    await client.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="LLM coherence scores",
        )
    )


async def _bootstrap_channels(guild: discord.Guild) -> None:
    """Create missing categories and channels. Idempotent."""
    existing_cats = {c.name: c for c in guild.categories}
    existing_chs = {ch.name for ch in guild.channels}

    for cat_name, channels in CHANNEL_LAYOUT.items():
        category = existing_cats.get(cat_name)
        if category is None:
            category = await guild.create_category(cat_name)
            log.info("Created category: %s", cat_name)

        for ch_name in channels:
            if ch_name not in existing_chs:
                await guild.create_text_channel(ch_name, category=category)
                log.info("Created #%s in %s", ch_name, cat_name)


@client.event
async def on_member_join(member: discord.Member) -> None:
    general = discord.utils.get(member.guild.text_channels, name="general")
    if general is None:
        return
    em = welcome_embed(member.display_name)
    await general.send(embed=em)


async def _pypi_version() -> str:
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(PYPI_URL)
        r.raise_for_status()
        return r.json()["info"]["version"]


async def _last_ci_run() -> tuple[str | None, str | None]:
    """Return (conclusion, html_url) for the latest CI run on main."""
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(
            f"{GH_API}/actions/workflows/ci.yml/runs",
            params={"branch": "main", "per_page": "1"},
        )
        if r.status_code != 200:
            return None, None
        runs = r.json().get("workflow_runs", [])
        if not runs:
            return None, None
        return runs[0].get("conclusion"), runs[0].get("html_url")


async def _latest_release() -> tuple[str, str, str]:
    """Return (tag, body, html_url) for the latest GitHub release."""
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(f"{GH_API}/releases/latest")
        r.raise_for_status()
        data = r.json()
        return data["tag_name"], data.get("body", ""), data["html_url"]


def register_commands(tree: app_commands.CommandTree) -> None:
    @tree.command(name="version", description="Current PyPI version")
    async def cmd_version(interaction: discord.Interaction) -> None:
        try:
            ver = await _pypi_version()
        except httpx.HTTPError:
            await interaction.response.send_message(
                "Failed to reach PyPI", ephemeral=True
            )
            return
        await interaction.response.send_message(
            f"**director-ai** `{ver}` · "
            f"[PyPI](https://pypi.org/project/director-ai/{ver}/)\n"
            f"```\npip install director-ai=={ver}\n```"
        )

    @tree.command(name="docs", description="Documentation links")
    async def cmd_docs(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "**Director-AI Documentation**\n"
            f"- [Quickstart]({DOCS_URL}/quickstart/)\n"
            f"- [Scorer API]({DOCS_URL}/reference/scorer/)\n"
            f"- [Streaming]({DOCS_URL}/reference/streaming/)\n"
            f"- [Configuration]({DOCS_URL}/reference/config/)\n"
            f"- [Deployment]({DOCS_URL}/deployment/production/)\n"
            f"- [Domain Presets]({DOCS_URL}/guide/presets/)\n"
            f"- [KB Ingestion]({DOCS_URL}/guide/kb-ingestion/)"
        )

    @tree.command(name="install", description="Installation options")
    async def cmd_install(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "**pip install**\n"
            "```bash\n"
            "pip install director-ai            # core (heuristic scoring)\n"
            "pip install director-ai[nli]       # + NLI model (DeBERTa)\n"
            "pip install director-ai[server]    # + REST API server\n"
            "pip install director-ai[vector]    # + vector store (ChromaDB)\n"
            'pip install "director-ai[nli,vector,server]"  # production\n'
            "```\n"
            "**Docker**\n"
            "```bash\n"
            "docker pull ghcr.io/anulum/director-ai:latest        # CPU\n"
            "docker pull ghcr.io/anulum/director-ai:latest-gpu    # GPU\n"
            "```"
        )

    @tree.command(name="status", description="Version + CI status")
    async def cmd_status(interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        try:
            ver = await _pypi_version()
        except httpx.HTTPError:
            ver = "unknown"
        conclusion, run_url = await _last_ci_run()
        em = status_embed(ver, conclusion, run_url)
        await interaction.followup.send(embed=em)

    @tree.command(name="quickstart", description="6-line guard snippet")
    async def cmd_quickstart(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "**6-line hallucination guard:**\n"
            "```python\n"
            "from director_ai import guard\n"
            "from openai import OpenAI\n"
            "\n"
            "client = guard(\n"
            "    OpenAI(),\n"
            '    facts={"policy": "Refunds within 30 days only"},\n'
            ")\n"
            "\n"
            "response = client.chat.completions.create(\n"
            '    model="gpt-4o-mini",\n'
            '    messages=[{"role": "user", "content": "What is the refund policy?"}],\n'
            ")\n"
            "```\n"
            f"[Full quickstart]({DOCS_URL}/quickstart/)"
        )

    @tree.command(name="profiles", description="Available domain presets")
    async def cmd_profiles(interaction: discord.Interaction) -> None:
        lines = [f"**`{name}`** — {desc}" for name, desc in PROFILES.items()]
        await interaction.response.send_message(
            "**Domain Presets**\n"
            + "\n".join(lines)
            + "\n\nUsage: `director-ai config --profile <name>`"
        )

    @tree.command(name="changelog", description="Latest release notes")
    async def cmd_changelog(interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        try:
            tag, body, url = await _latest_release()
        except httpx.HTTPError:
            await interaction.followup.send(
                "Failed to fetch release info", ephemeral=True
            )
            return
        # Truncate body to fit Discord's 4096-char embed limit
        desc = body[:1800] + "..." if len(body) > 1800 else body
        em = discord.Embed(
            title=f"Latest: {tag}",
            description=desc or "No release notes.",
            color=0x5865F2,
            url=url,
            timestamp=datetime.now(timezone.utc),
        )
        em.set_footer(text=REPO)
        await interaction.followup.send(embed=em)


client.run(BOT_TOKEN, log_handler=None)
