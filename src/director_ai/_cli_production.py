# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production scaffold validation CLI

"""Production scaffold validation for generated Director-AI deployments."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProductionScaffoldCheck:
    """One production-scaffold readiness check."""

    code: str
    passed: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        """Serialise the check for JSON reports."""

        return {
            "code": self.code,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass(frozen=True)
class ProductionScaffoldReport:
    """Aggregate production-scaffold validation report."""

    path: str
    require_secrets: bool
    checks: tuple[ProductionScaffoldCheck, ...]

    @property
    def passed(self) -> bool:
        """Return whether every scaffold check passed."""

        return all(check.passed for check in self.checks)

    @property
    def blockers(self) -> tuple[ProductionScaffoldCheck, ...]:
        """Return failed checks."""

        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, object]:
        """Serialise the validation report."""

        return {
            "path": self.path,
            "require_secrets": self.require_secrets,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "blockers": [check.to_dict() for check in self.blockers],
        }

    def to_cli_dict(self) -> dict[str, object]:
        """Serialise a redacted CLI report.

        The production check reads ``.env`` files to validate whether required
        secrets are present. Even though check messages never include secret
        values, the CLI keeps its machine-readable output to status, codes, and
        counts so scanners and shell logs cannot capture operator material.
        """

        return {
            "path": self.path,
            "require_secrets": self.require_secrets,
            "passed": self.passed,
            "check_count": len(self.checks),
            "blocker_count": len(self.blockers),
            "checks": [
                {"code": check.code, "passed": check.passed} for check in self.checks
            ],
            "blockers": [
                {"code": check.code, "passed": check.passed} for check in self.blockers
            ],
        }

    def to_markdown(self) -> str:
        """Return an operator-readable validation report."""

        lines = [
            "# Production Scaffold Check",
            "",
            f"path: {self.path}",
            f"require_secrets: {str(self.require_secrets).lower()}",
            f"passed: {str(self.passed).lower()}",
            "",
            "| Check | Result | Message |",
            "|---|---:|---|",
        ]
        for check in self.checks:
            result = "pass" if check.passed else "fail"
            lines.append(f"| {check.code} | {result} | {check.message} |")
        return "\n".join(lines) + "\n"

    def to_cli_markdown(self) -> str:
        """Return a redacted operator-readable validation report."""

        lines = [
            "# Production Scaffold Check",
            "",
            f"path: {self.path}",
            f"require_secrets: {str(self.require_secrets).lower()}",
            f"passed: {str(self.passed).lower()}",
            f"check_count: {len(self.checks)}",
            f"blocker_count: {len(self.blockers)}",
            "",
            "| Check | Result |",
            "|---|---:|",
        ]
        for check in self.checks:
            result = "pass" if check.passed else "fail"
            lines.append(f"| {check.code} | {result} |")
        return "\n".join(lines) + "\n"


REQUIRED_FILES = (
    "config.yaml",
    ".env",
    "docker-compose.yml",
    "facts.txt",
    "README.md",
    "monitoring/prometheus.yml",
    "secrets/README.md",
)

REQUIRED_DIRS = (
    "audit",
    "chroma",
    "monitoring",
    "secrets",
)

REQUIRED_ENV_KEYS = (
    "DIRECTOR_API_KEY_TENANT_MAP",
    "DIRECTOR_PROXY_API_KEYS",
    "DIRECTOR_LLM_API_URL",
    "DIRECTOR_UPSTREAM_URL",
    "DIRECTOR_KB_HMAC_KEYS",
    "DIRECTOR_CORS_ORIGINS",
)

REQUIRED_TRUE_CONFIG = (
    "production_mode",
    "tenant_routing",
    "coherence_require_model_backed_nli",
    "adaptive_threshold_fail_closed",
    "injection_detection_enabled",
    "injection_require_model_backed_nli",
    "injection_fail_closed_on_error",
    "sanitize_inputs",
    "redact_pii",
    "privacy_mode",
    "metrics_enabled",
    "metrics_require_auth",
    "review_queue_enabled",
    "knowledge_write_require_signature",
)

REQUIRED_CONFIG_VALUES = {
    "mode": "grounded",
    "stats_backend": "sqlite",
    "vector_backend": "chroma",
    "llm_provider": "local",
}

REQUIRED_COMPOSE_SNIPPETS = (
    'DIRECTOR_PRODUCTION_MODE: "true"',
    'DIRECTOR_USE_NLI: "true"',
    'DIRECTOR_METRICS_REQUIRE_AUTH: "true"',
    'DIRECTOR_KNOWLEDGE_WRITE_REQUIRE_SIGNATURE: "true"',
    'DIRECTOR_TENANT_ROUTING: "true"',
    'profiles: ["monitoring"]',
    "DIRECTOR_API_KEY_TENANT_MAP: ${DIRECTOR_API_KEY_TENANT_MAP:?",
    "DIRECTOR_LLM_API_URL: ${DIRECTOR_LLM_API_URL:?",
    "DIRECTOR_CORS_ORIGINS: ${DIRECTOR_CORS_ORIGINS:?",
    "--api-keys ${DIRECTOR_PROXY_API_KEYS:?",
    "--upstream-url ${DIRECTOR_UPSTREAM_URL:?",
    "director-ai[server,vector,nli,otel,presidio]",
)

FORBIDDEN_DEFAULT_SURFACE_SNIPPETS = (
    "meta_guard",
    "self_evolving",
    "continual_adversarial",
    'DIRECTOR_DRY_RUN: "true"',
)


def validate_production_scaffold(
    path: str | Path,
    *,
    require_secrets: bool = False,
) -> ProductionScaffoldReport:
    """Validate a generated production scaffold."""

    root = Path(path).resolve()
    checks: list[ProductionScaffoldCheck] = []

    checks.append(
        _check("root_exists", root.is_dir(), f"{root.as_posix()} exists"),
    )
    for relative in REQUIRED_FILES:
        target = root / relative
        checks.append(
            _check(
                f"file:{relative}",
                target.is_file(),
                f"{relative} is present",
            ),
        )
    for relative in REQUIRED_DIRS:
        target = root / relative
        checks.append(
            _check(
                f"dir:{relative}",
                target.is_dir(),
                f"{relative}/ is present",
            ),
        )

    config = _read_key_value_file(root / "config.yaml")
    checks.extend(_validate_config(config))

    env = _read_key_value_file(root / ".env")
    checks.extend(_validate_env(env, require_secrets=require_secrets))

    compose = _read_text(root / "docker-compose.yml")
    checks.extend(_validate_compose(compose))

    prometheus = _read_text(root / "monitoring" / "prometheus.yml")
    checks.extend(_validate_prometheus(prometheus))

    readme = _read_text(root / "README.md")
    checks.append(
        _check(
            "readme:auth_metrics",
            "Authorization: Bearer <api-key>" in readme
            and "docker compose --profile monitoring up" in readme,
            "README documents authenticated metrics and monitoring profile",
        ),
    )

    return ProductionScaffoldReport(
        path=root.as_posix(),
        require_secrets=require_secrets,
        checks=tuple(checks),
    )


def _validate_config(config: dict[str, str]) -> list[ProductionScaffoldCheck]:
    checks: list[ProductionScaffoldCheck] = []
    for key in REQUIRED_TRUE_CONFIG:
        checks.append(
            _check(
                f"config:{key}",
                _is_true(config.get(key, "")),
                f"config.yaml sets {key}: true",
            ),
        )
    for key, expected in REQUIRED_CONFIG_VALUES.items():
        checks.append(
            _check(
                f"config:{key}",
                config.get(key, "") == expected,
                f"config.yaml sets {key}: {expected}",
            ),
        )
    checks.append(
        _check(
            "config:no_dry_run",
            not _is_true(config.get("dry_run", "false")),
            "config.yaml does not enable dry_run",
        ),
    )
    return checks


def _validate_env(
    env: dict[str, str],
    *,
    require_secrets: bool,
) -> list[ProductionScaffoldCheck]:
    checks: list[ProductionScaffoldCheck] = []
    for key in REQUIRED_ENV_KEYS:
        checks.append(
            _check(
                f"env:{key}",
                key in env,
                f".env declares {key}",
            ),
        )
        if require_secrets:
            checks.append(
                _check(
                    f"secret:{key}",
                    bool(env.get(key, "").strip()),
                    f".env provides non-empty {key}",
                ),
            )
    if require_secrets:
        cors = env.get("DIRECTOR_CORS_ORIGINS", "").strip()
        checks.append(
            _check(
                "secret:cors_exact",
                cors.startswith("https://") and "*" not in cors,
                "DIRECTOR_CORS_ORIGINS is an exact HTTPS origin",
            ),
        )
        for key in ("DIRECTOR_LLM_API_URL", "DIRECTOR_UPSTREAM_URL"):
            value = env.get(key, "").strip()
            checks.append(
                _check(
                    f"secret:{key}:url",
                    value.startswith(("https://", "http://127.0.0.1")),
                    f"{key} points to HTTPS or localhost",
                ),
            )
    return checks


def _validate_compose(compose: str) -> list[ProductionScaffoldCheck]:
    checks: list[ProductionScaffoldCheck] = []
    for snippet in REQUIRED_COMPOSE_SNIPPETS:
        checks.append(
            _check(
                f"compose:{_slug(snippet)}",
                snippet in compose,
                f"docker-compose.yml contains {snippet}",
            ),
        )
    lowered = compose.lower()
    for snippet in FORBIDDEN_DEFAULT_SURFACE_SNIPPETS:
        checks.append(
            _check(
                f"compose:no_{_slug(snippet)}",
                snippet.lower() not in lowered,
                f"docker-compose.yml does not enable {snippet}",
            ),
        )
    return checks


def _validate_prometheus(prometheus: str) -> list[ProductionScaffoldCheck]:
    return [
        _check(
            "prometheus:auth_file",
            "credentials_file: /etc/prometheus/director-api-key" in prometheus,
            "Prometheus uses bearer credentials file",
        ),
        _check(
            "prometheus:director_target",
            "director-api:8000" in prometheus
            and "/v1/metrics/prometheus" in prometheus,
            "Prometheus scrapes authenticated Director metrics",
        ),
    ]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and (":" not in line or line.index("=") < line.index(":")):
            key, raw_value = line.split("=", 1)
        elif ":" in line:
            key, raw_value = line.split(":", 1)
        else:
            continue
        values[key.strip()] = _clean_scalar(raw_value)
    return values


def _clean_scalar(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:80]


def _check(code: str, passed: bool, message: str) -> ProductionScaffoldCheck:
    return ProductionScaffoldCheck(code=code, passed=passed, message=message)


def _cmd_production_check(args: list[str]) -> None:
    """Validate a generated production deployment scaffold."""

    path = Path("director_guard")
    require_secrets = False
    emit_json = False
    i = 0
    while i < len(args):
        if args[i] == "--path" and i + 1 < len(args):
            path = Path(args[i + 1])
            i += 2
        elif args[i] == "--require-secrets":
            require_secrets = True
            i += 1
        elif args[i] == "--json":
            emit_json = True
            i += 1
        else:
            print(
                "Usage: director-ai production-check "
                "[--path director_guard] [--require-secrets] [--json]",
            )
            sys.exit(1)

    report = validate_production_scaffold(path, require_secrets=require_secrets)
    if emit_json:
        print(json.dumps({"status": "completed", "details": "redacted"}))
    else:
        print("Production scaffold check completed. Details redacted; use exit code.")
    sys.exit(0 if report.passed else 1)
