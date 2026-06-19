# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLI Server/Proxy Commands
"""CLI subcommands for serving, proxy, and stress testing.

Extracted from cli.py to reduce module size.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from director_ai.core.config import DirectorConfig


def _apply_run_mode(config: DirectorConfig, run_mode: str) -> DirectorConfig:
    """Apply the explicit ``--dev`` / ``--production`` run mode to ``config``.

    Enforces that production is never entered implicitly: starting in production
    requires the explicit ``--production`` flag, which fails closed (with a clear
    message) when no effective API key is configured. Returns the resolved
    config; calls ``sys.exit(1)`` on a refusal.
    """
    from dataclasses import replace

    if run_mode == "production":
        try:
            config = replace(config, production_mode=True)
        except (ValueError, TypeError) as exc:
            print(f"Error: --production requires a hardened config: {exc}")
            print(
                "       Set DIRECTOR_API_KEYS (or DIRECTOR_API_KEY_TENANT_MAP) "
                "and DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS before --production."
            )
            sys.exit(1)
    elif run_mode == "dev":
        config = replace(config, production_mode=False)
    elif getattr(config, "production_mode", False):
        print(
            "Error: config resolves to production_mode but --production was not "
            "given. Refusing to start in production implicitly."
        )
        print("       Pass --production to confirm, or --dev for a dev server.")
        sys.exit(1)
    return config


def _resolve_bind_host(config: DirectorConfig, host: str, host_explicit: bool) -> str:
    """Resolve the server bind host from the ``--host`` flag and run mode.

    Precedence: an explicit ``--host`` wins; then an explicit
    ``DIRECTOR_SERVER_HOST`` environment variable (this is how the container
    image requests an all-interfaces bind, which is correct inside a container
    where exposure is controlled by the port mapping); then a production config
    keeps its configured/all-interfaces bind for a reverse-proxy deployment;
    otherwise a dev server binds to loopback so an unauthenticated ``serve --dev``
    is not exposed on every interface. Operators who want LAN access in dev opt
    in with ``--host 0.0.0.0`` or ``DIRECTOR_SERVER_HOST``.
    """
    if host_explicit:
        return host
    env_host = os.environ.get("DIRECTOR_SERVER_HOST", "").strip()
    if env_host:
        return env_host
    if getattr(config, "production_mode", False):
        # Production serves all interfaces by default for a reverse proxy, but
        # honours a host explicitly configured via profile (anything other than
        # the secure loopback default).
        configured = getattr(config, "server_host", "") or ""
        return "0.0.0.0" if configured in ("", "127.0.0.1") else configured  # nosec B104
    return "127.0.0.1"


def _cmd_serve(args: list[str]) -> None:
    port = 8080
    # Bind host is resolved after the run mode is known: explicit --host wins;
    # otherwise dev binds to loopback and production keeps its configured bind.
    host = ""
    host_explicit = False
    profile = "default"
    mode = ""
    workers = 1
    transport = "http"
    cors_origins = ""
    run_mode = ""  # "" (resolve from config), "dev", or "production"

    i = 0
    while i < len(args):
        if args[i] == "--production":
            run_mode = "production"
            i += 1
        elif args[i] == "--dev":
            run_mode = "dev"
            i += 1
        elif args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            if mode not in ("general", "grounded", "auto"):
                print(
                    f"Error: --mode must be 'general', 'grounded', or 'auto', got '{mode}'"
                )
                sys.exit(1)
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"Error: invalid port number: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            host_explicit = True
            i += 2
        elif args[i] == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
            i += 2
        elif args[i] == "--workers" and i + 1 < len(args):
            try:
                workers = int(args[i + 1])
                if workers < 1:
                    raise ValueError
            except ValueError:
                print(f"Error: invalid worker count: {args[i + 1]}")
                sys.exit(1)
            i += 2
        elif args[i] == "--cors-origins" and i + 1 < len(args):
            cors_origins = args[i + 1]
            i += 2
        elif args[i] == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
            if transport not in ("http", "grpc"):
                print(f"Error: --transport must be 'http' or 'grpc', got '{transport}'")
                sys.exit(1)
            i += 2
        else:
            i += 1

    from director_ai.core.config import DirectorConfig

    if profile != "default":
        config = DirectorConfig.from_profile(profile)
    else:
        config = DirectorConfig.from_env()
    if mode:
        config = DirectorConfig(**{**config.__dict__, "mode": mode})
    config = _apply_run_mode(config, run_mode)
    config.server_host = _resolve_bind_host(config, host, host_explicit)
    host = config.server_host
    config.server_port = port
    if cors_origins:
        config.cors_origins = cors_origins

    if transport == "grpc":
        from director_ai.grpc_server import create_grpc_server

        print(f"Starting Director AI gRPC server on port {port} (workers={workers})")
        server = create_grpc_server(config, max_workers=workers, port=port)
        server.start()
        server.wait_for_termination()
        return

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install director-ai[server]")
        sys.exit(1)

    from director_ai.server import create_app

    print(
        f"Starting Director AI server on {host}:{port} "
        f"(profile={config.profile}, workers={workers})",
    )

    if workers > 1:
        import os

        os.environ["DIRECTOR_PROFILE"] = profile
        os.environ["DIRECTOR_SERVER_HOST"] = host
        os.environ["DIRECTOR_SERVER_PORT"] = str(port)
        uvicorn.run(
            "director_ai.server:create_app",
            factory=True,
            host=host,
            port=port,
            workers=workers,
        )
    else:
        app = create_app(config)
        uvicorn.run(app, host=host, port=port)


def _cmd_proxy(args: list[str]) -> None:
    """Start OpenAI-compatible guardrail proxy."""
    port = 8080
    threshold = 0.6
    facts_path = None
    facts_root: str | None = None
    upstream_url = "https://api.openai.com"
    on_fail = "reject"
    api_keys: list[str] = []
    allow_http = False
    audit_db: str | None = None
    config_env = False

    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--threshold" and i + 1 < len(args):
            threshold = float(args[i + 1])
            i += 2
        elif args[i] == "--facts" and i + 1 < len(args):
            facts_path = args[i + 1]
            i += 2
        elif args[i] == "--facts-root" and i + 1 < len(args):
            facts_root = args[i + 1]
            i += 2
        elif args[i] == "--upstream-url" and i + 1 < len(args):
            upstream_url = args[i + 1]
            i += 2
        elif args[i] == "--on-fail" and i + 1 < len(args):
            on_fail = args[i + 1]
            if on_fail not in ("reject", "warn"):
                print(f"Error: --on-fail must be 'reject' or 'warn', got '{on_fail}'")
                sys.exit(1)
            i += 2
        elif args[i] == "--api-keys" and i + 1 < len(args):
            api_keys = [k.strip() for k in args[i + 1].split(",") if k.strip()]
            i += 2
        elif args[i] == "--allow-http-upstream":
            allow_http = True
            i += 1
        elif args[i] == "--audit-db" and i + 1 < len(args):
            audit_db = args[i + 1]
            i += 2
        elif args[i] == "--config-env":
            config_env = True
            i += 1
        else:
            i += 1

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install director-ai[server]")
        sys.exit(1)

    from director_ai.proxy import create_proxy_app

    config = None
    if config_env:
        from director_ai.core.config import DirectorConfig

        config = DirectorConfig.from_env()

    app = create_proxy_app(
        threshold=threshold,
        facts_path=facts_path,
        facts_root=facts_root,
        upstream_url=upstream_url,
        on_fail=on_fail,
        api_keys=api_keys or None,
        allow_http_upstream=allow_http,
        audit_db=audit_db,
        config=config,
    )

    print(
        f"Director-AI proxy on :{port} â†’ {upstream_url} "
        f"(threshold={threshold}, on_fail={on_fail})",
    )
    # CLI proxy bind; production deployments place it behind a reverse proxy.
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104


def _cmd_stress_test(args: list[str]) -> None:
    """Benchmark streaming kernel throughput."""
    import math
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    streams = 100
    tokens_per_stream = 50
    concurrency = 8
    json_output = False

    i = 0
    while i < len(args):
        if args[i] == "--streams" and i + 1 < len(args):
            streams = int(args[i + 1])
            i += 2
        elif args[i] == "--tokens-per-stream" and i + 1 < len(args):
            tokens_per_stream = int(args[i + 1])
            i += 2
        elif args[i] == "--concurrency" and i + 1 < len(args):
            concurrency = int(args[i + 1])
            i += 2
        elif args[i] == "--json":
            json_output = True
            i += 1
        else:
            i += 1

    from director_ai.core.runtime.streaming import StreamingKernel

    def _coherence_cb(token: str) -> float:
        h = hash(token) & 0xFFFFFFFF
        return 0.8 + 0.1 * math.sin(h)

    def _run_one(stream_id: int) -> dict[str, Any]:
        kernel = StreamingKernel()
        tokens = [f"tok{j}" for j in range(tokens_per_stream)]
        t0 = time.monotonic()
        session = kernel.stream_tokens(tokens, _coherence_cb)
        elapsed = time.monotonic() - t0
        return {
            "halted": session.halted,
            "tokens": session.token_count,
            "elapsed": elapsed,
        }

    latencies: list[float] = []
    halts = 0
    total_tokens = 0

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_one, sid) for sid in range(streams)]
        for future in as_completed(futures):
            result = future.result()
            latencies.append(result["elapsed"])
            if result["halted"]:
                halts += 1
            total_tokens += result["tokens"]
    t_total = max(time.monotonic() - t_start, 1e-9)

    latencies.sort()

    def _pct(sorted_vals: list[float], q: float) -> float:
        idx = int(q * (len(sorted_vals) - 1))
        return sorted_vals[idx]

    report = {
        "streams": streams,
        "tokens_per_stream": tokens_per_stream,
        "concurrency": concurrency,
        "total_seconds": round(t_total, 4),
        "streams_per_second": round(streams / t_total, 2),
        "tokens_per_second": round(total_tokens / t_total, 2),
        "halt_rate": round(halts / streams, 4),
        "latency_p50": round(_pct(latencies, 0.5), 6),
        "latency_p95": round(_pct(latencies, 0.95), 6),
        "latency_p99": round(_pct(latencies, 0.99), 6),
    }

    if json_output:
        print(json.dumps(report, indent=2))
    else:
        print(f"Streams:     {report['streams']}")
        print(f"Tokens/s:    {report['tokens_per_second']}")
        print(f"Streams/s:   {report['streams_per_second']}")
        print(f"Halt rate:   {report['halt_rate']:.2%}")
        print(f"Latency p50: {report['latency_p50'] * 1000:.2f}ms")
        print(f"Latency p95: {report['latency_p95'] * 1000:.2f}ms")
        print(f"Latency p99: {report['latency_p99'] * 1000:.2f}ms")
        print(f"Total time:  {report['total_seconds']:.2f}s")
