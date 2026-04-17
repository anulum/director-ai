# SPDX-License-Identifier: AGPL-3.0-or-later
.DEFAULT_GOAL := help
.PHONY: help test test-rust test-julia test-lean test-go test-all proto lint fmt docs docs-build bench clean build preflight preflight-fast bandit sast install-hooks docker-build docker-run backup julia-instantiate

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

test: ## Run Python tests with coverage
	pytest tests/ -v --cov=director_ai --cov-report=term --cov-fail-under=90

test-rust: ## Run Rust tests (backfire-kernel)
	cd backfire-kernel && cargo test --workspace

julia-instantiate: ## Install Julia tuner dependencies
	julia --project=tools/julia_tuner -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

test-julia: ## Run Julia threshold-tuner tests
	julia --project=tools/julia_tuner -e 'using Pkg; Pkg.test()'

test-lean: ## Build Lean 4 formal models (HaltMonitor)
	cd formal/HaltMonitor && lake build

test-go: ## Run Go gateway tests
	cd gateway/go && go test ./...

proto: ## Regenerate Python and Go stubs from schemas/proto/*.proto
	bash schemas/generate.sh

test-all: test test-rust test-julia test-lean test-go ## Run Python + Rust + Julia + Lean + Go checks

lint: ## Check style (ruff format + ruff check)
	ruff format --check src/ tests/
	ruff check src/ tests/

fmt: ## Auto-fix style
	ruff format src/ tests/
	ruff check --fix src/ tests/
	cd backfire-kernel && cargo fmt --all

bandit: ## SAST scan
	bandit -r src/director_ai/ -c pyproject.toml -q

sast: bandit ## Alias for bandit

preflight: ## Full preflight gate
	python tools/preflight.py

preflight-fast: ## Lint-only preflight (~5s)
	python tools/preflight.py --no-tests

docs: ## Local docs server
	mkdocs serve

docs-build: ## Build docs (strict)
	mkdocs build --strict

bench: ## Run regression benchmark suite
	python -m benchmarks.regression_suite

build: ## Build sdist + wheel
	python -m build

install-hooks: ## Install pre-push hook
	git config core.hooksPath .githooks
	@echo "Git hooks installed (.githooks/pre-push)"

docker-build: ## Build Docker image
	docker build -t director-ai:latest .

docker-run: ## Run Docker container
	docker run --rm -it -p 8080:8080 director-ai:latest

backup: ## Create git bundle backup
	@VERSION=$$(python -c "from director_ai import __version__; print(__version__)") && \
	DEST="../../.coordination/backups/director-ai-v$${VERSION}-stable-$$(date +%Y%m%d).bundle" && \
	git bundle create "$$DEST" --all && \
	echo "Backup: $$DEST ($$(du -h "$$DEST" | cut -f1))"

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	cd backfire-kernel && cargo clean
