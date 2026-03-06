# SPDX-License-Identifier: AGPL-3.0-or-later
.PHONY: test lint fmt docs bench clean build preflight bandit sast install-hooks docker-build docker-run backup

test:
	pytest tests/ -v --cov=director_ai --cov-report=term --cov-fail-under=90

test-rust:
	cd backfire-kernel && cargo test --workspace

test-all: test test-rust

lint:
	ruff format --check src/ tests/
	ruff check src/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/
	cd backfire-kernel && cargo fmt --all

bandit:
	bandit -r src/director_ai/ -c pyproject.toml -q

sast: bandit

preflight:
	python tools/preflight.py

preflight-fast:
	python tools/preflight.py --no-tests

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict

bench:
	python -m benchmarks.regression_suite

build:
	python -m build

install-hooks:
	git config core.hooksPath .githooks
	@echo "Git hooks installed (.githooks/pre-push)"

docker-build:
	docker build -t director-ai:latest .

docker-run:
	docker run --rm -it -p 8080:8080 director-ai:latest

backup:
	@VERSION=$$(python -c "from director_ai import __version__; print(__version__)") && \
	DEST="../../.coordination/backups/director-ai-v$${VERSION}-stable-$$(date +%Y%m%d).bundle" && \
	git bundle create "$$DEST" --all && \
	echo "Backup: $$DEST ($$(du -h "$$DEST" | cut -f1))"

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	cd backfire-kernel && cargo clean
