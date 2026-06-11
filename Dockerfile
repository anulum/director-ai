# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Production Docker Image
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────
#
# Build:
#   docker build -t director-ai .
#
# Run:
#   docker run -p 8080:8080 director-ai
#   docker run -p 8080:8080 -e DIRECTOR_USE_NLI=true director-ai
#
# Multi-stage build: builder (compile) + runtime (slim)
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ────────────────────────────────────────────────

FROM rust:1.95.0-slim@sha256:e14e87345b4d5964ddcc3491d27ee046a0f23820f340c3c1e24da6880141f7c0 AS rust-toolchain

FROM python:3.11-slim@sha256:d6e4d224f70f9e0172a06a3a2eba2f768eb146811a349278b38fff3a36463b47 AS builder

WORKDIR /build

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH="/usr/local/cargo/bin:${PATH}"

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential=12.12 ca-certificates=20250419 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src/ src/
COPY requirements/ requirements/
COPY backfire-kernel/ backfire-kernel/

ARG EXTRAS="server"
ARG REQUIREMENTS="requirements/docker-server.txt"
ARG BUILD_REQUIREMENTS="requirements/docker-build.txt"
RUN python -m pip install --no-cache-dir --require-hashes --no-deps --prefix=/install -r "$BUILD_REQUIREMENTS" \
    && python -m pip install --no-cache-dir --require-hashes --no-deps --prefix=/install -r "$REQUIREMENTS" \
    && PYTHONPATH=/install/lib/python3.11/site-packages PATH="/install/bin:${PATH}" \
        maturin build --release --manifest-path backfire-kernel/crates/backfire-ffi/Cargo.toml --out /tmp/backfire-wheel \
    && PYTHONPATH=/install/lib/python3.11/site-packages \
        python -m installer --prefix=/install /tmp/backfire-wheel/backfire_kernel-0.1.1-*.whl \
    && PYTHONPATH=/install/lib/python3.11/site-packages PATH="/install/bin:${PATH}" \
        python -m build --wheel --no-isolation --outdir /tmp/director-wheel . \
    && PYTHONPATH=/install/lib/python3.11/site-packages \
        python -m installer --prefix=/install /tmp/director-wheel/director_ai-*-py3-none-any.whl

# ── Stage 2: Runtime ────────────────────────────────────────────────

FROM python:3.11-slim@sha256:d6e4d224f70f9e0172a06a3a2eba2f768eb146811a349278b38fff3a36463b47

LABEL maintainer="Miroslav Sotek <protoscience@anulum.li>"
LABEL description="Director-AI — Real-time LLM hallucination guardrail"
LABEL org.opencontainers.image.source="https://github.com/anulum/director-ai"
LABEL org.opencontainers.image.license="Apache-2.0 AND BUSL-1.1"

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DIRECTOR_LOG_LEVEL=INFO \
    DIRECTOR_SERVER_HOST=0.0.0.0 \
    DIRECTOR_SERVER_PORT=8080

RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /app/director-models/_uploads \
    && chown -R appuser:appuser /app/director-models
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8080/v1/health'); r.raise_for_status()" || exit 1

ENTRYPOINT ["python", "-m", "director_ai.cli"]
CMD ["serve", "--port", "8080"]
