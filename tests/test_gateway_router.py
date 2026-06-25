"""
Tests for GatewayRouter — the pre-flight decision layer.
Verifies that routing, caching, and rate limiting all happen
BEFORE the orchestrator/agents are ever involved.
"""
import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.gateway.router import GatewayRouter, RouteDecision


@pytest.fixture
def mock_cache():
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)   # default: cache miss
    cache.set = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def router(mock_cache):
    return GatewayRouter(cache=mock_cache)


class TestGatewayRouterCacheHit:

    @pytest.mark.asyncio
    async def test_pipeline_cache_hit_returns_immediately(self, mock_cache):
        """Cache hit must return CACHE_HIT — no agent calls possible."""
        cached_payload = {
            "response": "Cached answer about LiteLLM",
            "sources": ["litellm.ai"],
            "blocked": False,
            "pipeline_stats": {"total_tokens": 500, "cache_hits": 0},
        }
        mock_cache.get = AsyncMock(return_value=cached_payload)
        router = GatewayRouter(cache=mock_cache)

        decision = await router.evaluate(
            query="What is LiteLLM?",
            user_id="user-1",
            request_id="req-1",
        )

        assert decision.route == RouteDecision.CACHE_HIT
        assert decision.cached_response is not None
        assert decision.cached_response["response"] == "Cached answer about LiteLLM"

    @pytest.mark.asyncio
    async def test_cache_miss_routes_to_agent_pipeline(self, router):
        """Cache miss must NOT return CACHE_HIT — let agents run."""
        decision = await router.evaluate(
            query="Explain RAGAS metrics in depth",
            user_id="user-2",
            request_id="req-2",
        )
        assert decision.route != RouteDecision.CACHE_HIT
        assert decision.cached_response is None

    @pytest.mark.asyncio
    async def test_cache_pipeline_response_stores_correctly(self, router, mock_cache):
        """Successful responses must be stored so next call hits cache."""
        response = {
            "response": "LiteLLM is a gateway.",
            "sources": [],
            "blocked": False,
            "pipeline_stats": {},
        }
        await router.cache_pipeline_response(query="What is LiteLLM?", response=response)
        mock_cache.set.assert_called_once()
        # Check the namespace used
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "pipeline_response"


class TestGatewayRouterRateLimit:

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_after_threshold(self, mock_cache):
        """Rate limiter must block before any agent is spawned."""
        router = GatewayRouter(cache=mock_cache)
        user_id = "heavy-user-rate-test"

        # Send requests up to the limit
        from app.gateway.router import _RATE_LIMIT_REQUESTS
        for _ in range(_RATE_LIMIT_REQUESTS):
            await router.evaluate(query="test query", user_id=user_id, request_id="r")

        # The next one should be rate limited
        decision = await router.evaluate(
            query="one more query",
            user_id=user_id,
            request_id="over-limit",
        )
        assert decision.route == RouteDecision.RATE_LIMITED
        assert decision.rate_limit_remaining == 0

    @pytest.mark.asyncio
    async def test_different_users_have_independent_limits(self, router):
        """Rate limits are per-user — one user's limit doesn't affect another."""
        from app.gateway.router import _RATE_LIMIT_REQUESTS

        user_a = "user-a-indep-test"
        user_b = "user-b-indep-test"

        for _ in range(_RATE_LIMIT_REQUESTS + 1):
            await router.evaluate("query", user_id=user_a, request_id="r")

        # user_b should still be fine
        decision_b = await router.evaluate("query", user_id=user_b, request_id="r")
        assert decision_b.route != RouteDecision.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_rate_limit_remaining_decreases(self, router):
        user_id = "user-countdown-test"
        d1 = await router.evaluate("q", user_id=user_id, request_id="r1")
        d2 = await router.evaluate("q", user_id=user_id, request_id="r2")
        assert d2.rate_limit_remaining < d1.rate_limit_remaining


class TestGatewayRouterModelSelection:

    @pytest.mark.asyncio
    async def test_short_simple_query_gets_fast_route(self, router):
        decision = await router.evaluate(
            query="What is AI?",
            user_id="u1",
            request_id="r1",
        )
        assert decision.route == RouteDecision.ROUTE_FAST
        assert decision.model_override is not None

    @pytest.mark.asyncio
    async def test_complex_query_gets_quality_or_standard_route(self, router):
        complex_query = (
            "Compare and contrast the architectural trade-offs between using LiteLLM "
            "versus building a custom LLM gateway in production, covering caching "
            "strategies, fallback mechanisms, cost optimization, and observability. "
            "Explain why each approach matters for a high-traffic agentic AI system."
        )
        decision = await router.evaluate(
            query=complex_query,
            user_id="u2",
            request_id="r2",
        )
        assert decision.route in (RouteDecision.ROUTE_QUALITY, RouteDecision.ROUTE_STANDARD)

    @pytest.mark.asyncio
    async def test_model_override_is_always_set_on_routed_request(self, router):
        """Agents must always receive a model from the gateway decision."""
        decision = await router.evaluate(
            query="Explain observability stacks",
            user_id="u3",
            request_id="r3",
        )
        if decision.route not in (RouteDecision.CACHE_HIT, RouteDecision.RATE_LIMITED):
            assert decision.model_override is not None
            assert len(decision.model_override) > 0

    @pytest.mark.asyncio
    async def test_enriched_query_is_populated(self, router):
        """Gateway must always return an enriched (at minimum cleaned) query."""
        decision = await router.evaluate(
            query="  What is RAGAS?  ",
            user_id="u4",
            request_id="r4",
        )
        assert decision.enriched_query.strip() != ""


class TestGatewayRouterCacheKey:

    def test_cache_key_is_normalised(self, router):
        """Minor whitespace/case differences must produce the same key."""
        k1 = router._make_pipeline_cache_key("What is LiteLLM?")
        k2 = router._make_pipeline_cache_key("what is litellm?")
        k3 = router._make_pipeline_cache_key("  What is LiteLLM?  ")
        # Case-normalised + stripped keys should all match
        assert k2 == k3   # whitespace stripped
        # k1 vs k2 differ only by case — both normalised to lowercase
        assert k1 == k2

    def test_different_queries_produce_different_keys(self, router):
        k1 = router._make_pipeline_cache_key("What is LiteLLM?")
        k2 = router._make_pipeline_cache_key("What is RAGAS?")
        assert k1 != k2


class TestGatewayRouterComplexityScoring:

    def test_short_query_low_complexity(self, router):
        score = router._estimate_complexity("What is AI?")
        assert score < 0.2

    def test_long_query_higher_complexity(self, router):
        long = " ".join(["complex technical term"] * 30)
        score = router._estimate_complexity(long)
        assert score >= 0.15

    def test_compare_keyword_raises_complexity(self, router):
        score = router._estimate_complexity(
            "Compare and contrast LiteLLM versus building a custom proxy"
        )
        assert score > router._estimate_complexity("What is LiteLLM?")
