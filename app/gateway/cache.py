"""
Caching Layer
=============
Two-level caching strategy:

1. Application-level cache (Redis) — exact key matching for tool results
2. Semantic cache (LiteLLM Proxy) — vector similarity for LLM responses

This module handles layer 1. Layer 2 is transparent via LiteLLM proxy config.
"""
import hashlib
import json
from typing import Any

import redis.asyncio as redis
import structlog

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class CacheClient:
    """
    Application-level Redis cache for tool results and intermediate agent outputs.

    Use this for:
    - Search tool results (expensive API calls)
    - Parsed/structured documents
    - Agent intermediate states

    Do NOT use for LLM responses — that's handled by LiteLLM semantic cache.
    """

    def __init__(self):
        self._client: redis.Redis | None = None

    async def _get_client(self) -> redis.Redis:
        if self._client is None:
            try:
                self._client = redis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                await self._client.ping()
                logger.info("cache.connected", url=settings.redis_url)
            except Exception as e:
                logger.warning("cache.unavailable", error=str(e))
                self._client = None
        return self._client

    def _make_key(self, namespace: str, data: Any) -> str:
        """Create a deterministic cache key from namespace + data."""
        content = json.dumps(data, sort_keys=True, default=str)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"agentic:{namespace}:{hash_val}"

    async def get(self, namespace: str, key_data: Any) -> Any | None:
        """Retrieve value from cache. Returns None on miss or error."""
        try:
            client = await self._get_client()
            if not client:
                return None
            key = self._make_key(namespace, key_data)
            value = await client.get(key)
            if value:
                logger.debug("cache.hit", namespace=namespace, key=key[:20])
                return json.loads(value)
            logger.debug("cache.miss", namespace=namespace, key=key[:20])
            return None
        except Exception as e:
            logger.warning("cache.get_error", error=str(e))
            return None

    async def set(
        self, namespace: str, key_data: Any, value: Any, ttl: int | None = None
    ) -> bool:
        """Store value in cache. Returns True on success."""
        try:
            client = await self._get_client()
            if not client:
                return False
            key = self._make_key(namespace, key_data)
            ttl = ttl or settings.cache_ttl_seconds
            await client.setex(key, ttl, json.dumps(value, default=str))
            logger.debug("cache.set", namespace=namespace, key=key[:20], ttl=ttl)
            return True
        except Exception as e:
            logger.warning("cache.set_error", error=str(e))
            return False

    async def delete(self, namespace: str, key_data: Any) -> bool:
        """Delete a cache entry."""
        try:
            client = await self._get_client()
            if not client:
                return False
            key = self._make_key(namespace, key_data)
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning("cache.delete_error", error=str(e))
            return False

    async def get_stats(self) -> dict:
        """Return Redis cache statistics."""
        try:
            client = await self._get_client()
            if not client:
                return {"status": "unavailable"}
            info = await client.info("stats")
            return {
                "status": "available",
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / max(
                        info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                    )
                ),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
