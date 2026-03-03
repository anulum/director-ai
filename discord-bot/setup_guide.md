# Director-AI Discord Bot — Setup Guide

## 1. Create the Discord Server

1. Open Discord -> "Add a Server" -> "Create My Own" -> name it **Director-AI**
2. Set server icon (use `docs/assets/header.png` or a cropped version)
3. Set server description: "Real-time LLM hallucination guardrail — NLI + RAG fact-checking with token-level streaming halt"

## 2. Create the Bot Application

1. Go to <https://discord.com/developers/applications>
2. Click **New Application** -> name it `Director-AI Bot`
3. Go to **Bot** tab -> click **Reset Token** -> copy the token
4. Save as `BOT_TOKEN` (you'll need it in step 5)
5. Under **Privileged Gateway Intents**, enable **Server Members Intent** (for welcome messages)

## 3. Invite the Bot

1. Go to **OAuth2 -> URL Generator**
2. Integration type: **Guild Install**
3. Scopes: `bot`, `applications.commands`
4. Bot Permissions: `Manage Channels`, `Send Messages`, `Embed Links`, `Read Message History`
5. Copy the generated URL -> open it -> select your server -> Authorize

## 4. Get the Guild (Server) ID

1. In Discord, enable **Developer Mode** (User Settings -> Advanced -> Developer Mode)
2. Right-click the server name -> **Copy Server ID**
3. Save as `GUILD_ID`

## 5. Create Discord Webhooks

Create two webhooks (Server Settings -> Integrations -> Webhooks):

| Webhook | Channel | GitHub Secret Name |
|---------|---------|-------------------|
| CI Status | `#ci-status` | `DISCORD_WEBHOOK_URL` |
| Announcements | `#announcements` | `DISCORD_ANNOUNCE_WEBHOOK` |

Copy each webhook URL for step 6.

## 6. Add GitHub Repository Secrets

Go to your repo -> Settings -> Secrets and variables -> Actions -> New repository secret:

| Secret | Value |
|--------|-------|
| `DISCORD_WEBHOOK_URL` | Webhook URL from `#ci-status` |
| `DISCORD_ANNOUNCE_WEBHOOK` | Webhook URL from `#announcements` |

## 7. GitHub Native Webhook (for #github-feed)

1. Create a third webhook in `#github-feed` channel, copy its URL
2. Go to GitHub repo -> Settings -> Webhooks -> Add webhook
3. Payload URL: `<webhook_url>/github` (append `/github` to the Discord URL)
4. Content type: `application/json`
5. Events: select **Pull requests**, **Issues**, **Pushes**

## 8. Run the Bot

```bash
cd discord-bot
pip install .
BOT_TOKEN=<your-token> GUILD_ID=<your-id> python bot.py
```

Or with Docker:

```bash
docker build -t director-ai-bot .
docker run -d --name director-bot \
  -e BOT_TOKEN=<your-token> \
  -e GUILD_ID=<your-id> \
  director-ai-bot
```

## 9. Channel Structure (auto-created)

The bot creates categories and channels on first start:

```
Development
  #ci-status         — CI pass/fail from GitHub Actions
  #github-feed       — PRs, issues, commits (GitHub native webhook)
Community
  #announcements     — release notes
  #general           — discussion + welcome messages
  #support           — user questions
```

## 10. Slash Commands

| Command | Description |
|---------|-------------|
| `/version` | Current PyPI version + install command |
| `/docs` | Documentation links (quickstart, API, streaming, deployment, presets, KB) |
| `/install` | pip + Docker install options |
| `/status` | PyPI version + last CI result |
| `/quickstart` | 6-line `guard()` code snippet |
| `/profiles` | All 10 domain presets with thresholds |
| `/changelog` | Latest GitHub release notes |

## 11. Bot Presence

The bot displays "Watching LLM coherence scores" as its activity status.

## 12. Welcome Messages

New members receive an embed in `#general` with quick-start links and slash command guide. Requires **Server Members Intent** enabled (step 2.5).
