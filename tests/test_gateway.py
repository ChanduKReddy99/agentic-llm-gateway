"""
Tests for Gateway Client and Cache
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.gateway.cache import CacheClient


class TestCacheClient:
    """Tests for the Redis cache client."""

    @pytest.fixture
    def cache(self):
        return CacheClient()

    def test_make_key_deterministic(self, cache):
        key1 = cache._make_key("search", {"query": "hello"})
        key2 = cache._make_key("search", {"query": "hello"})
        assert key1 == key2

    def test_make_key_different_for_different_data(self, cache):
        key1 = cache._make_key("search", {"query": "hello"})
        key2 = cache._make_key("search", {"query": "world"})
        assert key1 != key2

    def test_make_key_includes_namespace(self, cache):
        key1 = cache._make_key("search", {"query": "hello"})
        key2 = cache._make_key("tool", {"query": "hello"})
        assert key1 != key2
        assert "search" in key1
        assert "tool" in key2

    @pytest.mark.asyncio
    async def test_get_returns_none_on_unavailable_redis(self, cache):
        """When Redis is unavailable, get() should return None gracefully."""
        # Don't connect to real Redis in tests
        cache._client = None
        with patch.object(cache, '_get_client', return_value=None):
            result = await cache.get("search", {"query": "test"})
            assert result is None

    @pytest.mark.asyncio
    async def test_set_returns_false_on_unavailable_redis(self, cache):
        """When Redis is unavailable, set() should return False gracefully."""
        cache._client = None
        with patch.object(cache, '_get_client', return_value=None):
            result = await cache.set("search", {"query": "test"}, {"data": "value"})
            assert result is False

    @pytest.mark.asyncio
    async def test_get_stats_returns_unavailable(self, cache):
        """Stats should return unavailable status when Redis is down."""
        cache._client = None
        with patch.object(cache, '_get_client', return_value=None):
            stats = await cache.get_stats()
            assert stats["status"] in ("unavailable", "error")


class TestGatewayClient:
    """Tests for the LiteLLM gateway client."""

    def test_client_initialization(self):
        """Gateway client should initialize without errors."""
        from app.gateway.litellm_client import LiteLLMGatewayClient
        client = LiteLLMGatewayClient()
        assert client is not None
        assert client.client is not None  # OpenAI client pointed at LiteLLM proxy

    @pytest.mark.asyncio
    async def test_chat_completion_error_returns_graceful_response(self):
        """On error, gateway should return a graceful fallback dict."""
        from app.gateway.litellm_client import LiteLLMGatewayClient
        client = LiteLLMGatewayClient()

        # Mock the underlying OpenAI client to raise
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")
        client.client = mock_client

        result = await client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4o-mini",
            agent_name="test_agent",
        )

        assert "content" in result
        assert "error" in result
        assert result["cache_hit"] is False
        assert result["usage"]["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_health_check_returns_unhealthy_when_proxy_down(self):
        """Health check should return unhealthy status gracefully."""
        from app.gateway.litellm_client import LiteLLMGatewayClient
        client = LiteLLMGatewayClient()
        # Point to a port nothing is listening on
        client.settings = MagicMock()
        client.settings.litellm_proxy_url = "http://localhost:9999"
        client.settings.litellm_master_key = "test"

        health = await client.health_check()
        assert health["status"] == "unhealthy"
        assert "error" in health
