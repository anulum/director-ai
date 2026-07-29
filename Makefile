# SPDX-License-Identifier: Apache-2.0
.DEFAULT_GOAL := help
PYTHON ?= python
PYTHON_ONLY_CHECK_ARGS ?=
GENERATED_DOCS_DIR ?= $(CURDIR)/build/generated-docs
.PHONY: help test python-only-check test-rust test-julia test-lean test-go test-wasm test-all proto wasm-build lint fmt docs docs-build docs-all docs-polyglot docs-rust docs-go docs-typescript docs-julia docs-lean docs-protobuf bench clean build preflight preflight-fast bandit sast install-hooks docker-build docker-run backup julia-instantiate grpc-scoring ab-bench

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

test: ## Run Python tests with coverage
	pytest tests/ -v --cov=director_ai --cov-report=term --cov-fail-under=97

python-only-check: ## Run contributor checks without optional runtime toolchains
	$(PYTHON) tools/python_only_check.py $(PYTHON_ONLY_CHECK_ARGS)

test-rust: ## Run Rust tests (backfire-kernel)
	cd backfire-kernel && cargo test --workspace

julia-instantiate: ## Install Julia tuner dependencies
	julia --project=tools/julia_tuner -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

test-julia: ## Run Julia threshold-tuner tests
	julia --project=tools/julia_tuner -e 'using Pkg; Pkg.test()'

test-wasm: ## Build + test the backfire-wasm edge runtime
	cd backfire-kernel/crates/backfire-wasm && \
		CARGO_TARGET_DIR=$${CARGO_TARGET_DIR:-/media/anulum/GOTM/_caches/rust-target} \
		wasm-pack test --node

wasm-build: ## Build the backfire-wasm web module (pkg/)
	cd backfire-kernel/crates/backfire-wasm && \
		CARGO_TARGET_DIR=$${CARGO_TARGET_DIR:-/media/anulum/GOTM/_caches/rust-target} \
		wasm-pack build --target web --release

test-lean: ## Build Lean 4 formal models (HaltMonitor)
	cd formal/HaltMonitor && lake build

test-go: ## Run Go gateway tests
	cd gateway/go && go test ./...

proto: ## Regenerate Python and Go stubs from schemas/proto/*.proto
	bash schemas/generate.sh

grpc-scoring: ## Run the director.v1 CoherenceScoring gRPC server (port 50052)
	python -m director_ai.grpc_scoring --listen "[::]:50052"

ab-bench: ## A/B benchmark: gateway vs gateway+scoring (needs k6 installed)
	bash gateway/go/bench/ab_bench.sh

test-all: test test-rust test-julia test-lean test-go test-wasm ## Run Python + Rust + Julia + Lean + Go + WASM checks

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

reuse: ## Check SPDX/licence compliance (REUSE 3.3)
	reuse lint

sbom: ## Regenerate the per-extra declared-dependency SBOMs
	python -m scripts.generate_sboms

preflight: ## Full preflight gate
	python tools/preflight.py

preflight-fast: ## Lint-only preflight (~5s)
	python tools/preflight.py --no-tests

docs: ## Local docs server
	mkdocs serve

docs-build: ## Build docs (strict)
	mkdocs build --strict

docs-rust: ## Build warning-fatal Rust API documentation
	cd backfire-kernel && RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
	cd backfire-kernel && cargo test --doc --workspace --exclude backfire-ffi --locked

docs-go: ## Build static Go package documentation
	$(PYTHON) tools/build_go_docs.py --module-dir gateway/go --output "$(GENERATED_DOCS_DIR)/go"

docs-typescript: ## Build warning-fatal TypeScript API documentation
	cd packages/vercel-ai && npm ci --ignore-scripts && npm run docs -- --out "$(GENERATED_DOCS_DIR)/typescript"

docs-julia: ## Build warning-fatal Julia API documentation
	JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=tools/julia_tuner/docs -e 'using Pkg; Pkg.develop(PackageSpec(path="tools/julia_tuner")); Pkg.instantiate()'
	DIRECTOR_JULIA_DOCS_OUTPUT="$(GENERATED_DOCS_DIR)/julia" julia --project=tools/julia_tuner/docs tools/julia_tuner/docs/make.jl

docs-lean: ## Build Lean API documentation with doc-gen4
	cd formal/HaltMonitor/docbuild && DISABLE_EQUATIONS=1 lake build HaltMonitor:docs

docs-protobuf: ## Build Protobuf schema documentation
	mkdir -p "$(CURDIR)/build/tools/bin" "$(GENERATED_DOCS_DIR)/protobuf"
	GOBIN="$(CURDIR)/build/tools/bin" go install github.com/pseudomuto/protoc-gen-doc/cmd/protoc-gen-doc@v1.5.1
	PATH="$(CURDIR)/build/tools/bin:$$PATH" protoc \
		--doc_out="$(GENERATED_DOCS_DIR)/protobuf" \
		--doc_opt=html,index.html \
		proto/director.proto schemas/proto/director/v1/director.proto

docs-polyglot: docs-rust docs-go docs-typescript docs-julia docs-lean docs-protobuf ## Build every maintained non-Python API reference

docs-all: docs-build docs-polyglot ## Build the complete Python and polyglot documentation set

bench: ## Run regression benchmark suite
	python -m benchmarks.regression_suite

build: ## Build sdist + wheel
	python -m build

install-hooks: ## Install local git hooks
	git config core.hooksPath .githooks
	@echo "Git hooks installed (.githooks)"

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
