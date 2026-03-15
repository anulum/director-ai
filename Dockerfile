# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Production Docker Image
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
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

FROM python:3.11-slim@sha256:d6e4d224f70f9e0172a06a3a2eba2f768eb146811a349278b38fff3a36463b47 AS builder

WORKDIR /build

COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src/ src/
COPY requirements/ requirements/

ARG EXTRAS="server"
ARG REQUIREMENTS="requirements/docker-server.txt"
RUN pip install --no-cache-dir --require-hashes --no-deps --prefix=/install -r "$REQUIREMENTS" \
    && pip install --no-cache-dir --no-deps --prefix=/install .

# ── Stage 2: Runtime ────────────────────────────────────────────────

FROM python:3.11-slim@sha256:d6e4d224f70f9e0172a06a3a2eba2f768eb146811a349278b38fff3a36463b47

LABEL maintainer="Miroslav Sotek <protoscience@anulum.li>"
LABEL description="Director-AI — Real-time LLM hallucination guardrail"
LABEL org.opencontainers.image.source="https://github.com/anulum/director-ai"
LABEL org.opencontainers.image.license="AGPL-3.0-or-later"

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DIRECTOR_LOG_LEVEL=INFO \
    DIRECTOR_SERVER_HOST=0.0.0.0 \
    DIRECTOR_SERVER_PORT=8080

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8080/v1/health'); r.raise_for_status()" || exit 1

ENTRYPOINT ["python", "-m", "director_ai.cli"]
CMD ["serve", "--port", "8080"]
