# =============================================================
# Agentic AI + LLM Gateway — Makefile
# =============================================================

.PHONY: help install setup up down logs test demo ragas clean lint format

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────

install:  ## Install Python dependencies with uv
	uv sync

setup:  ## Full setup: copy .env, install deps
	@if [ ! -f .env ]; then cp .env.example .env && echo "✅ Created .env from .env.example — edit it with your API keys"; fi
	uv sync
	@echo "✅ Setup complete. Run 'make up' to start the observability stack."

# ─── Docker ──────────────────────────────────────────────────

up:  ## Start all Docker services (LiteLLM, Redis, Prometheus, Grafana, Loki, Tempo, Langfuse)
	docker-compose up -d
	@echo ""
	@echo "🚀 Services started!"
	@echo "   LiteLLM Proxy: http://localhost:4000"
	@echo "   Grafana:       http://localhost:3000  (admin/admin)"
	@echo "   Langfuse:      http://localhost:3001"
	@echo "   Prometheus:    http://localhost:9090"
	@echo ""
	@echo "Run 'make app' to start the FastAPI app."

down:  ## Stop all Docker services
	docker-compose down

restart:  ## Restart all Docker services
	docker-compose restart

logs:  ## Tail all Docker service logs
	docker-compose logs -f

logs-litellm:  ## Tail LiteLLM proxy logs
	docker-compose logs -f litellm

logs-grafana:  ## Tail Grafana logs
	docker-compose logs -f grafana

ps:  ## Show Docker service status
	docker-compose ps

# ─── Application ─────────────────────────────────────────────

app:  ## Run the FastAPI application (development mode)
	uv run python -m app.main

app-prod:  ## Run the FastAPI application (production mode)
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# ─── Scripts ─────────────────────────────────────────────────

setup-langfuse:  ## Register all prompts in local Langfuse (run once after make up)
	uv run python scripts/setup_langfuse.py

# ─── Prompt CI/CD ────────────────────────────────────────────

prompt-push:  ## Push local prompts/ → Langfuse as staging
	uv run python scripts/manage_prompts.py push --label staging

prompt-push-prod:  ## Push local prompts/ → Langfuse as production
	uv run python scripts/manage_prompts.py push --label production

prompt-pull:  ## Pull production prompts from Langfuse → local prompts/
	uv run python scripts/manage_prompts.py pull --label production

prompt-promote:  ## Promote staging → production in Langfuse
	uv run python scripts/manage_prompts.py promote --from-label staging --to-label production

prompt-diff:  ## Diff local prompts/ vs Langfuse production
	uv run python scripts/manage_prompts.py diff --label production

prompt-status:  ## Show all prompt versions in Langfuse
	uv run python scripts/manage_prompts.py status

demo:  ## Run the end-to-end demo script
	uv run python scripts/demo.py

ragas:  ## Run RAGAS batch evaluation
	uv run python scripts/run_ragas_eval.py

# ─── Tests ───────────────────────────────────────────────────

test:  ## Run all tests
	uv run pytest tests/ -v --tb=short

test-guardrails:  ## Run guardrails tests only
	uv run pytest tests/test_guardrails.py -v

test-ragas:  ## Run RAGAS tests only
	uv run pytest tests/test_ragas.py -v

test-agents:  ## Run agent tests only
	uv run pytest tests/test_agents.py -v

test-coverage:  ## Run tests with coverage report
	uv run pytest tests/ --cov=app --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

# ─── Code Quality ────────────────────────────────────────────

lint:  ## Run ruff linter
	uv run ruff check app/ tests/ scripts/

format:  ## Format code with ruff
	uv run ruff format app/ tests/ scripts/

# ─── Utilities ───────────────────────────────────────────────

health:  ## Check all service health endpoints
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "❌ App not running"
	@curl -s http://localhost:4000/health | python3 -m json.tool || echo "❌ LiteLLM not running"

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -f ragas_eval_results.csv
	rm -rf htmlcov/ .coverage

# ─── Quick Start ─────────────────────────────────────────────

quickstart: setup up  ## One-command quickstart: setup + start Docker
	@echo ""
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo ""
	@echo "✅ Ready! Now:"
	@echo "   1. Edit .env with your API keys"
	@echo "   2. Run: make app"
	@echo "   3. Visit: http://localhost:8000/docs"
	@echo "   4. Run demo: make demo"
