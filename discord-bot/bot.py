# (C) 1998-2026 Miroslav Sotek. All rights reserved.

from __future__ import annotations

import logging
import os
import sys

import discord
import httpx
from discord import app_commands

from embeds import status_embed

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

CHANNELS = ["announcements", "ci-status", "github-feed", "support", "general"]


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
    existing = {ch.name for ch in guild.channels}
    for name in CHANNELS:
        if name not in existing:
            await guild.create_text_channel(name)
            log.info("Created #%s", name)


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


def register_commands(tree: app_commands.CommandTree) -> None:
    @tree.command(name="version", description="Current PyPI version")
    async def cmd_version(interaction: discord.Interaction) -> None:
        try:
            ver = await _pypi_version()
        except httpx.HTTPError:
            await interaction.response.send_message("Failed to reach PyPI", ephemeral=True)
            return
        await interaction.response.send_message(
            f"**director-ai** `{ver}`\n```\npip install director-ai=={ver}\n```"
        )

    @tree.command(name="docs", description="Documentation links")
    async def cmd_docs(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "**Director-AI Docs**\n"
            "- [Quickstart](https://anulum.github.io/director-ai/quickstart/)\n"
            "- [API Reference](https://anulum.github.io/director-ai/reference/scorer/)\n"
            "- [Streaming](https://anulum.github.io/director-ai/guide/streaming/)\n"
            "- [Deployment](https://anulum.github.io/director-ai/deployment/production/)"
        )

    @tree.command(name="install", description="Installation options")
    async def cmd_install(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "```bash\n"
            "pip install director-ai            # core\n"
            "pip install director-ai[nli]       # + NLI models\n"
            "pip install director-ai[server]    # + REST server\n"
            "pip install director-ai[vector]    # + vector store\n"
            "```\n"
            "Docker:\n"
            "```bash\n"
            "docker pull ghcr.io/anulum/director-ai:latest\n"
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

    @tree.command(name="quickstart", description="Quick setup snippet")
    async def cmd_quickstart(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "```python\n"
            "from director_ai import CoherenceAgent\n"
            "\n"
            "agent = CoherenceAgent()\n"
            "result = agent.review(\n"
            '    prompt="What causes tides?",\n'
            '    response="Tides are caused by the Moon\'s gravity.",\n'
            ")\n"
            "print(result.accepted, result.score)\n"
            "```\n"
            "[Full quickstart](https://anulum.github.io/director-ai/quickstart/)"
        )


client.run(BOT_TOKEN, log_handler=None)
