# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quickstart project scaffolding for the director-ai CLI."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from director_ai.core.config import DirectorConfig

_VALID_PROFILES = (
    "fast",
    "lite",
    "rules",
    "embed",
    "thorough",
    "research",
    "medical",
    "finance",
    "legal",
    "creative",
    "customer_support",
    "summarization",
    "production",
)


def _write_quickstart_config(
    out_dir: Path,
    profile: str,
    config: DirectorConfig,
) -> None:
    coherence_threshold = config.coherence_threshold
    hard_limit = config.hard_limit
    use_nli = config.use_nli
    lines = [
        f"# Director-AI configuration - profile: {profile}\n"
        f"coherence_threshold: {coherence_threshold}\n",
        f"hard_limit: {hard_limit}\n",
        f"use_nli: {str(use_nli).lower()}\n",
        f"profile: {profile}\n",
    ]
    if profile == "production":
        lines.extend(
            [
                "production_mode: true\n",
                "mode: grounded\n",
                "tenant_routing: true\n",
                "coherence_require_model_backed_nli: true\n",
                "adaptive_threshold_fail_closed: true\n",
                "injection_detection_enabled: true\n",
                "injection_require_model_backed_nli: true\n",
                "injection_fail_closed_on_error: true\n",
                "sanitize_inputs: true\n",
                "redact_pii: true\n",
                "privacy_mode: true\n",
                "metrics_enabled: true\n",
                "metrics_require_auth: true\n",
                "rate_limit_rpm: 120\n",
                "review_queue_enabled: true\n",
                "knowledge_write_require_signature: true\n",
                "knowledge_write_hmac_keys: ''\n",
                "audit_log_path: audit/audit.jsonl\n",
                "compliance_db_path: audit/compliance.sqlite\n",
                "feedback_db_path: audit/feedback.sqlite\n",
                "stats_backend: sqlite\n",
                "stats_db_path: audit/stats.sqlite\n",
                "vector_backend: chroma\n",
                "chroma_persist_dir: chroma\n",
                "api_key_tenant_map: ''\n",
                "llm_provider: local\n",
                "llm_api_url: http://127.0.0.1:8081/v1\n",
            ],
        )
    (out_dir / "config.yaml").write_text("".join(lines), encoding="utf-8")


def _write_quickstart_facts(out_dir: Path) -> None:
    (out_dir / "facts.txt").write_text(
        "The sky is blue due to Rayleigh scattering.\n"
        "Water boils at 100 degrees Celsius at sea level.\n"
        "The Earth orbits the Sun once every 365.25 days.\n",
        encoding="utf-8",
    )


def _write_quickstart_guard(out_dir: Path) -> None:
    (out_dir / "guard.py").write_text(
        '"""Minimal Director-AI guard - run: python guard.py"""\n'
        "from pathlib import Path\n"
        "\n"
        "from director_ai.core import CoherenceScorer, GroundTruthStore\n"
        "from director_ai.core.config import DirectorConfig\n"
        "\n"
        "_HERE = Path(__file__).resolve().parent\n"
        "config = DirectorConfig.from_yaml(str(_HERE / 'config.yaml'))\n"
        "store = GroundTruthStore()\n"
        "with open(_HERE / 'facts.txt') as f:\n"
        "    for line in f:\n"
        "        line = line.strip()\n"
        "        if line:\n"
        "            store.add(line[:20], line)\n"
        "\n"
        "scorer = CoherenceScorer(\n"
        "    threshold=config.coherence_threshold,\n"
        "    ground_truth_store=store,\n"
        "    use_nli=config.use_nli,\n"
        ")\n"
        "\n"
        "approved, score = scorer.review(\n"
        '    "What color is the sky?", "The sky is blue."\n'
        ")\n"
        'print(f"Approved: {approved}  Score: {score.score:.3f}")\n',
        encoding="utf-8",
    )


def _write_quickstart_env(out_dir: Path, profile: str, threshold: float) -> None:
    if profile == "production":
        env_text = (
            "# Required before `docker compose up`:\n"
            '# DIRECTOR_API_KEY_TENANT_MAP={"director-service-key":"tenant-default"}\n'
            "# DIRECTOR_PROXY_API_KEYS=director-service-key\n"
            "# DIRECTOR_LLM_API_URL=https://llm-gateway.internal/v1\n"
            "# DIRECTOR_UPSTREAM_URL=https://llm-gateway.internal\n"
            "# DIRECTOR_KB_HMAC_KEYS=key-id:hex-or-random-secret\n"
            "# DIRECTOR_CORS_ORIGINS=https://console.example.com\n"
            "DIRECTOR_API_KEY_TENANT_MAP=\n"
            "DIRECTOR_PROXY_API_KEYS=\n"
            "DIRECTOR_LLM_API_URL=\n"
            "DIRECTOR_UPSTREAM_URL=\n"
            "DIRECTOR_KB_HMAC_KEYS=\n"
            "DIRECTOR_CORS_ORIGINS=\n"
            "DIRECTOR_TENANT_ID=tenant-default\n"
            f"DIRECTOR_COHERENCE_THRESHOLD={threshold}\n"
            "DIRECTOR_SERVER_PORT=8000\n"
        )
        (out_dir / ".env").write_text(env_text, encoding="utf-8")
        return

    env_text = (
        f"DIRECTOR_QUICKSTART_PROFILE={profile}\n"
        f"DIRECTOR_COHERENCE_THRESHOLD={threshold}\n"
        "DIRECTOR_HARD_LIMIT=0.2\n"
        "DIRECTOR_SOFT_LIMIT=0.35\n"
        "DIRECTOR_MODE=auto\n"
        "DIRECTOR_USE_NLI=false\n"
        "DIRECTOR_SCORER_BACKEND=auto\n"
        "DIRECTOR_VECTOR_BACKEND=chroma\n"
        "DIRECTOR_CHROMA_PERSIST_DIR=/data/chroma\n"
        "DIRECTOR_CHROMA_COLLECTION=director_quickstart\n"
        "DIRECTOR_METRICS_ENABLED=true\n"
        "DIRECTOR_LOG_LEVEL=INFO\n"
        "DIRECTOR_SERVER_HOST=0.0.0.0\n"
        "DIRECTOR_SERVER_PORT=8000\n"
    )
    (out_dir / ".env").write_text(env_text, encoding="utf-8")


def _quickstart_proxy_service(threshold: float) -> str:
    return (
        "  director-proxy:\n"
        "    image: python:3.12.11-slim\n"
        "    working_dir: /app\n"
        "    ports:\n"
        '      - "8080:8080"\n'
        "    volumes:\n"
        "      - ./facts.txt:/app/facts.txt:ro\n"
        "    command:\n"
        "      - /bin/sh\n"
        "      - -lc\n"
        "      - >-\n"
        "        pip install --no-cache-dir 'director-ai[server]'\n"
        "        && director-ai proxy --port 8080 --facts /app/facts.txt\n"
        f"        --facts-root /app --threshold {threshold} --on-fail reject\n"
    )


def _quickstart_api_service() -> str:
    return (
        "  director-api:\n"
        "    image: python:3.12.11-slim\n"
        "    working_dir: /app\n"
        "    ports:\n"
        '      - "8000:8000"\n'
        "    env_file: .env\n"
        "    volumes:\n"
        "      - ./facts.txt:/app/facts.txt:ro\n"
        "      - ./chroma:/data/chroma\n"
        "    command:\n"
        "      - /bin/sh\n"
        "      - -lc\n"
        "      - >-\n"
        "        pip install --no-cache-dir 'director-ai[server,vector]'\n"
        "        && director-ai ingest /app/facts.txt --persist /data/chroma\n"
        "        && director-ai serve --host 0.0.0.0 --port 8000\n"
    )


def _quickstart_onnx_service(threshold: float) -> str:
    return (
        "  director-proxy-onnx:\n"
        '    profiles: ["onnx"]\n'
        "    image: python:3.12.11-slim\n"
        "    working_dir: /app\n"
        "    ports:\n"
        '      - "8081:8080"\n'
        "    env_file: .env\n"
        "    environment:\n"
        '      DIRECTOR_USE_NLI: "true"\n'
        "      DIRECTOR_SCORER_BACKEND: onnx\n"
        "      DIRECTOR_ONNX_PATH: /models/factcg-onnx\n"
        "    volumes:\n"
        "      - ./facts.txt:/app/facts.txt:ro\n"
        "      - ./chroma:/data/chroma\n"
        "      - ./models/factcg-onnx:/models/factcg-onnx:ro\n"
        "    command:\n"
        "      - /bin/sh\n"
        "      - -lc\n"
        "      - >-\n"
        "        pip install --no-cache-dir 'director-ai[server,vector,nli,onnx]'\n"
        "        && director-ai proxy --port 8080 --facts /app/facts.txt\n"
        f"        --facts-root /app --threshold {threshold} --on-fail reject --config-env\n"
    )


def _quickstart_production_api_service() -> str:
    return (
        "  director-api:\n"
        "    image: python:3.12.11-slim\n"
        "    working_dir: /app\n"
        "    ports:\n"
        '      - "8000:8000"\n'
        "    env_file: .env\n"
        "    environment:\n"
        '      DIRECTOR_PRODUCTION_MODE: "true"\n'
        '      DIRECTOR_USE_NLI: "true"\n'
        '      DIRECTOR_COHERENCE_REQUIRE_MODEL_BACKED_NLI: "true"\n'
        '      DIRECTOR_ADAPTIVE_THRESHOLD_FAIL_CLOSED: "true"\n'
        '      DIRECTOR_INJECTION_DETECTION_ENABLED: "true"\n'
        '      DIRECTOR_INJECTION_REQUIRE_MODEL_BACKED_NLI: "true"\n'
        '      DIRECTOR_INJECTION_FAIL_CLOSED_ON_ERROR: "true"\n'
        '      DIRECTOR_SANITIZE_INPUTS: "true"\n'
        '      DIRECTOR_REDACT_PII: "true"\n'
        '      DIRECTOR_PRIVACY_MODE: "true"\n'
        "      DIRECTOR_API_KEY_TENANT_MAP: ${DIRECTOR_API_KEY_TENANT_MAP:?Set DIRECTOR_API_KEY_TENANT_MAP in .env}\n"
        "      DIRECTOR_LLM_API_URL: ${DIRECTOR_LLM_API_URL:?Set DIRECTOR_LLM_API_URL in .env}\n"
        "      DIRECTOR_CORS_ORIGINS: ${DIRECTOR_CORS_ORIGINS:?Set DIRECTOR_CORS_ORIGINS in .env}\n"
        "      DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS: ${DIRECTOR_KB_HMAC_KEYS:?Set DIRECTOR_KB_HMAC_KEYS in .env}\n"
        "      DIRECTOR_LLM_PROVIDER: local\n"
        "      DIRECTOR_MODE: grounded\n"
        "      DIRECTOR_VECTOR_BACKEND: chroma\n"
        "      DIRECTOR_CHROMA_PERSIST_DIR: /data/chroma\n"
        "      DIRECTOR_CHROMA_COLLECTION: director_production\n"
        '      DIRECTOR_TENANT_ROUTING: "true"\n'
        '      DIRECTOR_KNOWLEDGE_WRITE_REQUIRE_SIGNATURE: "true"\n'
        '      DIRECTOR_METRICS_ENABLED: "true"\n'
        '      DIRECTOR_METRICS_REQUIRE_AUTH: "true"\n'
        '      DIRECTOR_RATE_LIMIT_RPM: "120"\n'
        '      DIRECTOR_REVIEW_QUEUE_ENABLED: "true"\n'
        "      DIRECTOR_AUDIT_LOG_PATH: /data/audit/audit.jsonl\n"
        "      DIRECTOR_COMPLIANCE_DB_PATH: /data/audit/compliance.sqlite\n"
        "      DIRECTOR_FEEDBACK_DB_PATH: /data/audit/feedback.sqlite\n"
        "      DIRECTOR_STATS_BACKEND: sqlite\n"
        "      DIRECTOR_STATS_DB_PATH: /data/audit/stats.sqlite\n"
        '      DIRECTOR_LOG_JSON: "true"\n'
        '      DIRECTOR_OTEL_ENABLED: "true"\n'
        "    volumes:\n"
        "      - ./facts.txt:/app/facts.txt:ro\n"
        "      - ./chroma:/data/chroma\n"
        "      - ./audit:/data/audit\n"
        "    command:\n"
        "      - /bin/sh\n"
        "      - -lc\n"
        "      - >-\n"
        "        pip install --no-cache-dir 'director-ai[server,vector,nli,otel,presidio]'\n"
        "        && director-ai ingest /app/facts.txt --persist /data/chroma\n"
        "        && director-ai serve --host 0.0.0.0 --port 8000\n"
        "    healthcheck:\n"
        '      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen(\'http://localhost:8000/v1/ready\')"]\n'
        "      interval: 15s\n"
        "      timeout: 5s\n"
        "      retries: 5\n"
        "    restart: unless-stopped\n"
    )


def _quickstart_production_proxy_service(threshold: float) -> str:
    return (
        "  director-proxy:\n"
        "    image: python:3.12.11-slim\n"
        "    working_dir: /app\n"
        "    ports:\n"
        '      - "8080:8080"\n'
        "    env_file: .env\n"
        "    environment:\n"
        "      DIRECTOR_API_KEY_TENANT_MAP: ${DIRECTOR_API_KEY_TENANT_MAP:?Set DIRECTOR_API_KEY_TENANT_MAP in .env}\n"
        "      DIRECTOR_LLM_API_URL: ${DIRECTOR_LLM_API_URL:?Set DIRECTOR_LLM_API_URL in .env}\n"
        "    volumes:\n"
        "      - ./facts.txt:/app/facts.txt:ro\n"
        "    command:\n"
        "      - /bin/sh\n"
        "      - -lc\n"
        "      - >-\n"
        "        pip install --no-cache-dir 'director-ai[server,nli]'\n"
        "        && director-ai proxy --port 8080 --facts /app/facts.txt\n"
        f"        --facts-root /app --threshold {threshold} --on-fail reject\n"
        "        --api-keys ${DIRECTOR_PROXY_API_KEYS:?Set DIRECTOR_PROXY_API_KEYS in .env}\n"
        "        --upstream-url ${DIRECTOR_UPSTREAM_URL:?Set DIRECTOR_UPSTREAM_URL in .env}\n"
        "        --config-env\n"
        "    restart: unless-stopped\n"
    )


def _write_quickstart_prometheus(out_dir: Path) -> None:
    monitoring_dir = out_dir / "monitoring"
    monitoring_dir.mkdir()
    (monitoring_dir / "prometheus.yml").write_text(
        "global:\n"
        "  scrape_interval: 15s\n"
        "scrape_configs:\n"
        "  - job_name: director-api\n"
        "    metrics_path: /v1/metrics/prometheus\n"
        "    authorization:\n"
        "      credentials_file: /etc/prometheus/director-api-key\n"
        "    static_configs:\n"
        "      - targets: ['director-api:8000']\n",
        encoding="utf-8",
    )
    secrets_dir = out_dir / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "README.md").write_text(
        "# Production secrets\n"
        "\n"
        "Create `director-api-key` with an API key present in "
        "`DIRECTOR_API_KEY_TENANT_MAP` before running the monitoring profile.\n",
        encoding="utf-8",
    )


def _quickstart_prometheus_service() -> str:
    return (
        "  prometheus:\n"
        '    profiles: ["monitoring"]\n'
        "    image: prom/prometheus:v2.54.1\n"
        "    ports:\n"
        '      - "9090:9090"\n'
        "    volumes:\n"
        "      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro\n"
        "      - ./secrets/director-api-key:/etc/prometheus/director-api-key:ro\n"
        "    depends_on:\n"
        "      - director-api\n"
        "    restart: unless-stopped\n"
    )


def _write_quickstart_compose(
    out_dir: Path,
    threshold: float,
    profile: str,
) -> None:
    (out_dir / "chroma").mkdir()
    if profile == "production":
        (out_dir / "audit").mkdir()
        _write_quickstart_prometheus(out_dir)
        (out_dir / "docker-compose.yml").write_text(
            "services:\n"
            f"{_quickstart_production_api_service()}"
            f"{_quickstart_production_proxy_service(threshold)}"
            f"{_quickstart_prometheus_service()}",
            encoding="utf-8",
        )
        return

    onnx_dir = out_dir / "models" / "factcg-onnx"
    onnx_dir.mkdir(parents=True)
    (onnx_dir / "README.md").write_text(
        "# FactCG ONNX Model Directory\n"
        "\n"
        "Place the exported ONNX model files here before running:\n"
        "\n"
        "```bash\n"
        "director-ai export --format onnx --output models/factcg-onnx\n"
        "```\n"
        "\n"
        "Expected files: `model.onnx`, `config.json`, `tokenizer.json`, "
        "`tokenizer_config.json`, and `special_tokens_map.json`.\n"
        "\n"
        "```bash\n"
        "docker compose --profile onnx up director-proxy-onnx\n"
        "```\n",
        encoding="utf-8",
    )
    (out_dir / "docker-compose.yml").write_text(
        "services:\n"
        f"{_quickstart_proxy_service(threshold)}"
        f"{_quickstart_api_service()}"
        f"{_quickstart_onnx_service(threshold)}",
        encoding="utf-8",
    )


def _write_quickstart_readme(
    out_dir: Path,
    profile: str,
    include_compose: bool,
) -> None:
    compose_block = ""
    if include_compose:
        if profile == "production":
            compose_block = (
                "\n"
                "## Docker Compose\n"
                "\n"
                "Fill `.env` with `DIRECTOR_API_KEY_TENANT_MAP`, "
                "`DIRECTOR_PROXY_API_KEYS`, `DIRECTOR_LLM_API_URL`, "
                "`DIRECTOR_UPSTREAM_URL`, "
                "`DIRECTOR_KB_HMAC_KEYS`, and `DIRECTOR_CORS_ORIGINS` before "
                "starting the stack.\n"
                "\n"
                "```bash\n"
                "docker compose up\n"
                'curl -H "Authorization: Bearer <api-key>" '
                "http://localhost:8000/v1/metrics/prometheus\n"
                "```\n"
                "\n"
                "For Prometheus, write the same API key to "
                "`secrets/director-api-key`, then run:\n"
                "\n"
                "```bash\n"
                "docker compose --profile monitoring up\n"
                "```\n"
            )
        else:
            compose_block = (
                "\n"
                "## Docker Compose\n"
                "\n"
                "```bash\n"
                "docker compose up\n"
                "curl http://localhost:8080/health\n"
                "curl http://localhost:8000/v1/health\n"
                "```\n"
                "\n"
                "`director-proxy` exposes the chat guard on port 8080.\n"
                "`director-api` exposes the FastAPI service on port 8000 with\n"
                "local Chroma persistence in `./chroma`.\n"
                "\n"
                "For FactCG ONNX, put exported model files in\n"
                "`models/factcg-onnx/` and run:\n"
                "\n"
                "```bash\n"
                "docker compose --profile onnx up director-proxy-onnx\n"
                "```\n"
            )
    (out_dir / "README.md").write_text(
        f"# Director-AI Guard (profile: {profile})\n"
        "\n"
        "```bash\n"
        "pip install director-ai\n"
        "python guard.py\n"
        "```\n"
        "\n"
        "Edit `facts.txt` to add your own knowledge base.\n"
        "Edit `config.yaml` to tune thresholds.\n"
        f"{compose_block}",
        encoding="utf-8",
    )


def _run_quickstart_compose(out_dir: Path) -> None:
    docker = shutil.which("docker")
    if docker is None:
        print("Error: docker compose is required for --run.")
        sys.exit(1)
    try:
        # Fixed argv, no shell, and an explicit generated quickstart directory.
        subprocess.run(  # nosec B603
            [docker, "compose", "up"],
            cwd=out_dir,
            check=True,
        )
    except FileNotFoundError:
        print("Error: docker compose is required for --run.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Error: docker compose failed with exit code {exc.returncode}.")
        sys.exit(exc.returncode)


def _load_profile_for_scaffold(profile: str) -> DirectorConfig:
    """Load a profile config for scaffolding, tolerating production fail-close.

    Production-mode profiles fail closed without secrets, but the quickstart
    scaffold only needs the profile's template defaults — the generated ``.env``
    and ``docker-compose.yml`` require the operator to supply real secrets via
    shell interpolation. When the profile fails to load, retry with placeholder
    secrets that are used only to materialise the in-memory config and never
    written into the scaffold.
    """
    import os

    from director_ai.core.config import DirectorConfig

    try:
        return DirectorConfig.from_profile(profile)
    except ValueError:
        placeholders = {
            "DIRECTOR_API_KEYS": "scaffold-placeholder-key",
            "DIRECTOR_API_KEY_TENANT_MAP": (
                '{"scaffold-placeholder-key":"scaffold-tenant"}'
            ),
            "DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS": (
                '{"scaffold":"scaffold-placeholder-signing-secret-32x"}'
            ),
        }
        saved = {key: os.environ.get(key) for key in placeholders}
        try:
            os.environ.update(placeholders)
            return DirectorConfig.from_profile(profile)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _cmd_quickstart(args: list[str]) -> None:
    """Scaffold a working director-ai project in one command."""
    if args and args[0] in ("-h", "--help", "help"):
        _print_quickstart_help()
        return

    profile = "fast"
    include_compose = True
    run_compose = False
    i = 0
    while i < len(args):
        if args[i] == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
            i += 2
        elif args[i] == "--no-compose":
            include_compose = False
            i += 1
        elif args[i] == "--run":
            run_compose = True
            i += 1
        else:
            i += 1

    if profile not in _VALID_PROFILES:
        print(f"Unknown profile '{profile}'. Choose from: {', '.join(_VALID_PROFILES)}")
        sys.exit(1)

    out_dir = Path("director_guard")
    if out_dir.exists():
        print(f"Error: {out_dir}/ already exists. Remove it or use a new dir.")
        sys.exit(1)

    cfg = _load_profile_for_scaffold(profile)
    out_dir.mkdir()

    _write_quickstart_config(
        out_dir,
        profile,
        cfg,
    )
    _write_quickstart_facts(out_dir)
    _write_quickstart_guard(out_dir)
    if include_compose:
        _write_quickstart_env(out_dir, profile, cfg.coherence_threshold)
        _write_quickstart_compose(out_dir, cfg.coherence_threshold, profile)
    _write_quickstart_readme(out_dir, profile, include_compose)

    print(f"Created {out_dir}/ - run: python {out_dir}/guard.py")
    if include_compose:
        print(f"Compose quickstart: cd {out_dir} && docker compose up")
    if run_compose:
        _run_quickstart_compose(out_dir)


def _print_quickstart_help() -> None:
    """Print quickstart options without creating scaffold files."""
    print(
        "Usage: director-ai quickstart [options]\n"
        "\n"
        "Scaffold a local director-ai project with config, facts, and guard script.\n"
        "\n"
        "Options:\n"
        "  --profile NAME         Built-in profile to scaffold (default: fast)\n"
        "  --no-compose           Omit docker-compose.yml and .env\n"
        "  --run                  Run docker compose after scaffolding\n"
    )
