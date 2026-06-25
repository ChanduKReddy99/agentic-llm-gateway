"""
FastAPI Application Entry Point
================================
Production-grade FastAPI app with:
- Full observability (metrics, traces, structured logs)
- LiteLLM gateway integration
- Agentic pipeline endpoints
- Health & admin endpoints
"""
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from app.agents.orchestrator import AgentOrchestrator
from app.config.settings import get_settings
from app.observability.logging_config import setup_logging
from app.observability.tracing import setup_tracing

# Setup logging and tracing FIRST
setup_logging()
logger = structlog.get_logger(__name__)
settings = get_settings()

# Global orchestrator instance
_orchestrator: AgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global _orchestrator

    logger.info(
        "app.starting",
        env=settings.app_env,
        host=settings.app_host,
        port=settings.app_port,
    )

    # Initialize OpenTelemetry tracing
    setup_tracing()

    # Initialize the agent orchestrator
    _orchestrator = AgentOrchestrator()

    logger.info("app.ready", message="Agentic AI Gateway is ready to serve requests 🚀")

    yield

    # Shutdown
    logger.info("app.shutdown", message="Shutting down gracefully")
    if _orchestrator:
        _orchestrator.langfuse.flush()


# ─── Create FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic AI + LLM Gateway",
    description=(
        "Production-grade agentic AI with LiteLLM gateway, guardrails, "
        "full observability stack. RAGAS evaluation runs offline (dev/staging only)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus FastAPI instrumentation
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
).instrument(app).expose(app, endpoint="/internal/metrics-fastapi")

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ─── Request Logging Middleware ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    structlog.contextvars.clear_contextvars()

    response = await call_next(request)

    duration = time.time() - start
    logger.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response


# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The question or task for the agentic pipeline",
        examples=["What are the benefits of using an LLM gateway like LiteLLM?"],
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for session tracking in Langfuse",
    )


class QueryResponse(BaseModel):
    response: str
    sources: list[str] = []
    blocked: bool = False
    blocked_at: str = ""
    violations: list[dict] = []
    pipeline_stats: dict = {}
    request_id: str = ""
    trace_id: str = ""


class HealthResponse(BaseModel):
    status: str
    services: dict


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with links to key services."""
    return {
        "app": "Agentic AI + LLM Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /api/v1/query",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "gateway_health": "GET /api/v1/gateway/health",
            "gateway_spend": "GET /api/v1/gateway/spend",
        },
        "observability": {
            "grafana": "http://localhost:3000",
            "langfuse": "http://localhost:3001",
            "prometheus": "http://localhost:9090",
        },
    }


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Agentic Pipeline"])
async def process_query(request: QueryRequest):
    """
    Process a query through the full 2-agent pipeline.
    
    Flow:
      1. Input guardrails (PII, injection, length)
      2. Research Agent (search + LLM synthesis)
      3. Synthesis Agent (draft + self-critique)
      4. Output guardrails (toxicity, PII leakage)
      5. RAGAS evaluation — runs OFFLINE, not here (see scripts/run_ragas_eval.py)
    
    All LLM calls go through LiteLLM gateway (caching + fallbacks).
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    result = await _orchestrator.process(
        query=request.query,
        user_id=request.user_id,
    )

    return QueryResponse(**result)


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """
    Comprehensive health check for all services.
    Used by load balancers and monitoring systems.
    """
    if not _orchestrator:
        return HealthResponse(
            status="starting",
            services={"orchestrator": "not_ready"},
        )

    gateway_health = await _orchestrator.gateway.health_check()
    cache_stats = await _orchestrator.cache.get_stats()

    all_healthy = (
        gateway_health.get("status") == "healthy"
        # Don't fail health check if Redis is down — it's optional
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services={
            "orchestrator": "healthy",
            "litellm_gateway": gateway_health,
            "redis_cache": cache_stats,
            "research_agent": "ready",
            "synthesis_agent": "ready",
        },
    )


@app.get("/api/v1/gateway/health", tags=["Gateway"])
async def gateway_health():
    """Check LiteLLM proxy health and available models."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Not ready")
    health = await _orchestrator.gateway.health_check()
    models = await _orchestrator.gateway.get_proxy_models()
    return {"health": health, "available_models": models}


@app.get("/api/v1/gateway/spend", tags=["Gateway"])
async def gateway_spend():
    """Get LLM spend statistics from LiteLLM proxy."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Not ready")
    return await _orchestrator.gateway.get_spend_stats()


@app.get("/api/v1/gateway/cost", tags=["Gateway"])
async def gateway_cost_breakdown():
    """
    Per-agent cost breakdown — answers: which agent costs the most?

    How LiteLLM calculates this:
      Every agent call passes through the LiteLLM proxy.
      The proxy reads token counts from each LLM response,
      multiplies by the model price table, and logs to Langfuse.
      This endpoint shows the Prometheus-side rollup.

    To see per-call detail: open Langfuse at http://localhost:3001
    Filter by tag agent_name=research_agent or synthesis_agent.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Not ready")

    # LiteLLM proxy spend logs (what the proxy tracked)
    proxy_spend = await _orchestrator.gateway.get_spend_stats()

    return {
        "note": (
            "LiteLLM proxy tracks cost per call automatically. "
            "Every agent LLM call passes through the proxy which reads "
            "token counts and multiplies by model price table. "
            "See Langfuse at http://localhost:3001 for per-call breakdown."
        ),
        "how_it_works": {
            "step_1": "Agent calls chat_completion(model=X, agent_name=Y)",
            "step_2": "Request hits LiteLLM proxy",
            "step_3": "Proxy checks Redis cache — HIT: cost=/bin/sh, MISS: forward to LLM",
            "step_4": "LLM returns response with usage.prompt_tokens + usage.completion_tokens",
            "step_5": "Proxy looks up model price: gpt-4o-mini input=/bin/sh.00015/1K output=/bin/sh.0006/1K",
            "step_6": "cost = (prompt_tokens/1K * input_price) + (completion_tokens/1K * output_price)",
            "step_7": "Proxy logs {model, tokens, cost, agent_name} to Langfuse callback",
            "step_8": "litellm_client.py also emits cost to Prometheus for Grafana",
        },
        "proxy_spend_logs": proxy_spend,
    }


@app.get("/api/v1/cache/stats", tags=["Cache"])
async def cache_stats():
    """Get Redis cache hit/miss statistics."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Not ready")
    return await _orchestrator.cache.get_stats()


@app.post("/api/v1/eval/ragas", tags=["Evaluation (Dev Only)"])
async def run_ragas_eval(
    question: str,
    answer: str,
    contexts: list[str],
):
    """
    On-demand RAGAS evaluation — DEV / STAGING only.

    This endpoint is for development and CI use.
    RAGAS is NOT called inline during production request handling.
    For batch evaluation run: scripts/run_ragas_eval.py
    """
    if settings.app_env == "production":
        raise HTTPException(
            status_code=403,
            detail="RAGAS evaluation endpoint is disabled in production. Run scripts/run_ragas_eval.py offline.",
        )
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Not ready")
    from app.ragas_eval.evaluator import RAGASEvaluator
    scores = await RAGASEvaluator().evaluate(
        question=question,
        answer=answer,
        contexts=contexts,
    )
    return {"ragas_scores": scores, "note": "Dev/staging eval — not run in production pipeline"}


@app.get("/api/v1/settings", tags=["Operations"])
async def get_app_settings():
    """Return non-sensitive application settings (for debugging)."""
    return {
        "env": settings.app_env,
        "models": {
            "research_agent": settings.research_agent_model,
            "synthesis_agent": settings.synthesis_agent_model,
        },
        "guardrails": {
            "input_enabled": settings.enable_input_guardrails,
            "output_enabled": settings.enable_output_guardrails,
            "pii_detection": settings.pii_detection_enabled,
        },
        # RAGAS runs offline only — see scripts/run_ragas_eval.py
        "cache_ttl_seconds": settings.cache_ttl_seconds,
    }


# ─── Main Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
