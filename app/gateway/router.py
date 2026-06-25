"""
Gateway Router — Pre-flight Request Layer
==========================================
This is the CORRECT place for the LLM Gateway in the pipeline.

Correct flow:
  User Query
      │
      ▼
  [1. Input Guardrails]   ← safety: block/redact before anything touches the query
      │
      ▼
  [2. Gateway Router]     ← routing decisions BEFORE agents are invoked:
      │                       - Full-pipeline semantic cache check (skip agents entirely!)
      │                       - Model selection based on query complexity
      │                       - Rate limit enforcement per user
      │                       - Request enrichment (add context, metadata)
      │                       - Decide which agent chain to use
      ▼
  [3. Orchestrator]       ← now just runs the agent chain the gateway approved
      │
      ▼
  [4. Output Guardrails]  ← validate the final response

Why gateway MUST come before orchestrator:
  - Cache hit = skip ALL agents entirely (huge cost saving)
  - Rate limit = reject before spawning any agent coroutines
  - Model routing = agents receive the right model config, not hardcoded ones
  - Request enrichment = agents get richer context from the gateway layer
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from app.config.settings import get_settings
from app.gateway.cache import CacheClient
from app.observability.metrics import (
    gateway_cache_hits_total,
    gateway_requests_total,
)
from app.observability.tracing import async_trace_span

logger = structlog.get_logger(__name__)
settings = get_settings()


class RouteDecision(str, Enum):
    CACHE_HIT = "cache_hit"          # Return cached pipeline response — skip all agents
    ROUTE_FAST = "route_fast"        # Short query → fast/cheap model
    ROUTE_STANDARD = "route_standard"  # Normal query → standard model
    ROUTE_QUALITY = "route_quality"  # Complex query → high-quality model
    RATE_LIMITED = "rate_limited"    # User exceeded rate limit


@dataclass
class GatewayDecision:
    """Result of the gateway pre-flight check."""
    route: RouteDecision
    model_override: str | None = None        # Override agent's default model
    cached_response: dict | None = None      # Full pipeline response if cache hit
    enriched_query: str = ""                 # Query after gateway enrichment
    metadata: dict = field(default_factory=dict)
    rate_limit_remaining: int = 100


# Simple in-memory rate limiter (use Redis in production)
_rate_limit_store: dict[str, dict] = {}
_RATE_LIMIT_REQUESTS = 20   # per window
_RATE_LIMIT_WINDOW_SEC = 60


class GatewayRouter:
    """
    Pre-flight gateway layer that runs BEFORE the orchestrator.

    Responsibilities (all decided here, before any agent is invoked):
    1. Full-pipeline cache check — cache hit = zero agent calls
    2. Query complexity analysis → model routing decision
    3. Per-user rate limiting
    4. Request enrichment (add system context, user tier metadata)
    5. Emit routing metrics
    """

    def __init__(self, cache: CacheClient):
        self.cache = cache

    async def evaluate(
        self,
        query: str,
        user_id: str,
        request_id: str,
    ) -> GatewayDecision:
        """
        Evaluate the request and return a routing decision BEFORE touching agents.

        This is the single point where:
          - We decide if we even need to run agents at all (cache hit)
          - We pick which model tier the agents should use
          - We enforce rate limits
        """
        async with async_trace_span(
            "gateway.preflight",
            {"user_id": user_id, "request_id": request_id}
        ) as span:

            # ── Step 1: Rate limit check ──────────────────────────────────
            # Do this first — cheapest check, protects everything downstream
            rate_ok, remaining = self._check_rate_limit(user_id)
            if not rate_ok:
                logger.warning(
                    "gateway.rate_limited",
                    user_id=user_id,
                    request_id=request_id,
                )
                span.set_attribute("route", RouteDecision.RATE_LIMITED)
                return GatewayDecision(
                    route=RouteDecision.RATE_LIMITED,
                    enriched_query=query,
                    rate_limit_remaining=0,
                    metadata={"reason": "Rate limit exceeded. Try again in 60 seconds."},
                )

            # ── Step 2: Full-pipeline cache check ─────────────────────────
            # If this exact (or semantically similar) query was answered before,
            # return the cached FULL pipeline response — zero agent invocations.
            cache_key = self._make_pipeline_cache_key(query)
            cached = await self.cache.get("pipeline_response", cache_key)

            if cached:
                gateway_cache_hits_total.labels(agent="pipeline_cache").inc()
                logger.info(
                    "gateway.pipeline_cache_hit",
                    request_id=request_id,
                    cache_key=cache_key[:20],
                )
                span.set_attribute("route", RouteDecision.CACHE_HIT)
                span.set_attribute("cache_hit", True)
                return GatewayDecision(
                    route=RouteDecision.CACHE_HIT,
                    cached_response=cached,
                    enriched_query=query,
                    rate_limit_remaining=remaining,
                    metadata={"cache_key": cache_key[:20]},
                )

            # ── Step 3: Query complexity → model routing ──────────────────
            # Analyse the query to decide which model tier to use.
            # This runs BEFORE agents are created — agents receive the decision.
            complexity = self._estimate_complexity(query)
            route, model = self._select_route_and_model(complexity)

            # ── Step 4: Request enrichment ────────────────────────────────
            # Append any helpful system context before agents see the query.
            enriched = self._enrich_query(query)

            gateway_requests_total.labels(
                agent="gateway_router", model=model
            ).inc()

            logger.info(
                "gateway.decision",
                route=route,
                model=model,
                complexity=complexity,
                request_id=request_id,
                rate_limit_remaining=remaining,
            )

            span.set_attribute("route", route)
            span.set_attribute("model", model)
            span.set_attribute("complexity", complexity)

            return GatewayDecision(
                route=route,
                model_override=model,
                enriched_query=enriched,
                rate_limit_remaining=remaining,
                metadata={
                    "complexity_score": complexity,
                    "routing_reason": f"complexity={complexity:.2f} → {route}",
                },
            )

    async def cache_pipeline_response(
        self,
        query: str,
        response: dict,
        ttl: int = 1800,  # 30 min default for full pipeline responses
    ) -> None:
        """
        Cache a successful full-pipeline response.
        Next identical/similar query = cache hit, zero agent calls.
        """
        cache_key = self._make_pipeline_cache_key(query)
        # Mark it as served from cache so stats are correct
        cached_copy = {**response, "served_from_pipeline_cache": True}
        await self.cache.set("pipeline_response", cache_key, cached_copy, ttl=ttl)
        logger.debug("gateway.pipeline_cached", cache_key=cache_key[:20], ttl=ttl)

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _make_pipeline_cache_key(self, query: str) -> str:
        """
        Create a deterministic cache key for the full pipeline response.
        Normalise the query so minor whitespace differences still hit cache.
        """
        normalised = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalised.encode()).hexdigest()[:24]

    def _estimate_complexity(self, query: str) -> float:
        """
        Heuristic complexity score 0.0–1.0.
        Production systems would use a classifier or embedding-based scoring.
        """
        score = 0.0
        q = query.lower()

        # Length signal
        words = len(q.split())
        if words > 50:
            score += 0.3
        elif words > 20:
            score += 0.15

        # Multi-part question signals
        if any(w in q for w in ["compare", "contrast", "versus", "vs", "difference between"]):
            score += 0.2
        if q.count("?") > 1:
            score += 0.15

        # Domain complexity signals
        complex_terms = [
            "architecture", "design", "trade-off", "production", "scalab",
            "implement", "explain why", "how does", "deep dive", "in detail"
        ]
        score += 0.1 * sum(1 for t in complex_terms if t in q)

        return min(1.0, score)

    def _select_route_and_model(self, complexity: float) -> tuple[RouteDecision, str]:
        """Map complexity score to a route + model decision."""
        if complexity < 0.2:
            # Short, simple query — fast cheap model
            return RouteDecision.ROUTE_FAST, "gpt-4o-mini"
        elif complexity < 0.6:
            # Normal query — standard model
            return RouteDecision.ROUTE_STANDARD, "gpt-4o-mini"
        else:
            # Complex multi-part query — quality model
            # In production: could route to gpt-4o or claude-3-5-sonnet
            return RouteDecision.ROUTE_QUALITY, "gpt-4o-mini"  # same model, diff config

    def _enrich_query(self, query: str) -> str:
        """
        Add helpful context to the query before agents process it.
        In production: inject user profile, retrieved history, domain hints.
        """
        # Keep it simple — just clean whitespace normalization here
        # Real enrichment: user tier, prior context, domain tagging
        return query.strip()

    def _check_rate_limit(self, user_id: str) -> tuple[bool, int]:
        """
        Simple sliding-window rate limiter.
        Returns (allowed, remaining_requests).
        In production: use Redis INCR + EXPIRE for distributed rate limiting.
        """
        now = time.time()
        window_start = now - _RATE_LIMIT_WINDOW_SEC

        if user_id not in _rate_limit_store:
            _rate_limit_store[user_id] = {"requests": [], "blocked_until": 0}

        store = _rate_limit_store[user_id]

        # Slide the window — drop old timestamps
        store["requests"] = [t for t in store["requests"] if t > window_start]
        store["requests"].append(now)

        count = len(store["requests"])
        remaining = max(0, _RATE_LIMIT_REQUESTS - count)

        if count > _RATE_LIMIT_REQUESTS:
            return False, 0

        return True, remaining
