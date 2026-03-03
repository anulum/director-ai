# Director-AI Discord Bot — Setup Guide

## 1. Create the Discord Server

1. Open Discord → "Add a Server" → "Create My Own" → name it **Director-AI**
2. Enable **Community** features (Server Settings → Enable Community) — required for announcement channels

## 2. Create the Bot Application

1. Go to <https://discord.com/developers/applications>
2. Click **New Application** → name it `Director-AI Bot`
3. Go to **Bot** tab → click **Reset Token** → copy the token
4. Save as `BOT_TOKEN` (you'll need it in step 5)
5. Under **Privileged Gateway Intents**, leave all unchecked (not needed)

## 3. Invite the Bot

1. Go to **OAuth2 → URL Generator**
2. Scopes: `bot`, `applications.commands`
3. Bot Permissions: `Manage Channels`, `Send Messages`, `Embed Links`, `Read Message History`
4. Copy the generated URL → open it → select your server → Authorize

## 4. Get the Guild (Server) ID

1. In Discord, enable **Developer Mode** (User Settings → Advanced → Developer Mode)
2. Right-click the server name → **Copy Server ID**
3. Save as `GUILD_ID`

## 5. Create Discord Webhooks

Create two webhooks (Server Settings → Integrations → Webhooks):

| Webhook | Channel | GitHub Secret Name |
|---------|---------|-------------------|
| CI Status | `#ci-status` | `DISCORD_WEBHOOK_URL` |
| Announcements | `#announcements` | `DISCORD_ANNOUNCE_WEBHOOK` |

Copy each webhook URL for step 6.

## 6. Add GitHub Repository Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
|--------|-------|
| `DISCORD_WEBHOOK_URL` | Webhook URL from `#ci-status` |
| `DISCORD_ANNOUNCE_WEBHOOK` | Webhook URL from `#announcements` |

## 7. GitHub Native Webhook (for #github-feed)

1. Create a third webhook in `#github-feed` channel, copy its URL
2. Go to GitHub repo → Settings → Webhooks → Add webhook
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

On first start the bot creates any missing channels (`#announcements`, `#ci-status`, `#github-feed`, `#support`, `#general`).

## 9. Verify

- `/version` — returns current PyPI version
- `/docs` — returns documentation links
- `/install` — returns install commands
- `/status` — returns version + last CI result
- `/quickstart` — returns a code snippet
- Push to `main` → `#ci-status` receives a CI embed
- Create a release → `#announcements` receives a release embed
