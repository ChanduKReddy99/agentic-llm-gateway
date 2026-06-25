"""
Search Tool — Research Agent's primary tool
===========================================
Uses Tavily Search API (https://tavily.com) for real web search.
Results are cached in Redis to avoid redundant API calls.

Falls back to a minimal stub if TAVILY_API_KEY is not configured.
"""
import asyncio
import time
from typing import Any

import structlog

from app.config.settings import get_settings
from app.gateway.cache import CacheClient
from app.observability.metrics import tool_calls_total, tool_latency_seconds
from app.observability.tracing import async_trace_span

logger = structlog.get_logger(__name__)
settings = get_settings()


def _get_tavily_client():
    """Return an async Tavily client, or None if key is missing."""
    if not settings.tavily_api_key:
        logger.warning(
            "search_tool.tavily_missing",
            message="TAVILY_API_KEY not set — search will return stub results",
        )
        return None
    try:
        from tavily import AsyncTavilyClient
        return AsyncTavilyClient(api_key=settings.tavily_api_key)
    except ImportError:
        logger.warning(
            "search_tool.tavily_import_error",
            message="tavily-python not installed. Run: uv sync",
        )
        return None


class SearchTool:
    """
    Web search tool used by the Research Agent.
    Calls Tavily Search API for real-time web results.
    Results are cached in Redis for 1 hour to reduce API calls.
    """

    def __init__(self, cache: CacheClient):
        self.cache = cache
        self._client = _get_tavily_client()

    async def search(self, query: str, max_results: int = 3) -> dict:
        """
        Execute a real web search via Tavily and return snippets + sources.
        Results are cached for 1 hour.
        """
        async with async_trace_span("tool.search", {"query": query[:100]}):
            start = time.time()
            cache_key = {"query": query.lower().strip(), "max_results": max_results}

            # Check Redis cache first
            cached = await self.cache.get("search", cache_key)
            if cached:
                logger.info("tool.search.cache_hit", query=query[:50])
                tool_calls_total.labels(tool_name="search", status="cache_hit").inc()
                return cached

            # Call Tavily or fall back to stub
            if self._client:
                result = await self._search_tavily(query, max_results)
            else:
                result = self._stub_result(query)

            # Cache for 1 hour
            await self.cache.set("search", cache_key, result, ttl=3600)

            elapsed = time.time() - start
            tool_latency_seconds.labels(tool_name="search").observe(elapsed)
            tool_calls_total.labels(tool_name="search", status="success").inc()

            logger.info(
                "tool.search.completed",
                query=query[:50],
                results=len(result.get("results", [])),
                source="tavily" if self._client else "stub",
                latency_ms=round(elapsed * 1000, 2),
            )

            return result

    async def _search_tavily(self, query: str, max_results: int) -> dict:
        """Call the Tavily Search API and normalise the response."""
        try:
            response: Any = await self._client.search(
                query=query,
                search_depth="advanced",   # advanced = more thorough results
                max_results=max_results,
                include_answer=False,       # we want raw search results only
                include_raw_content=False,
            )

            results = []
            for item in response.get("results", [])[:max_results]:
                results.append({
                    "snippet": item.get("content", ""),
                    "source": item.get("url", ""),
                    "title": item.get("title", ""),
                    "relevance_score": round(item.get("score", 0.9), 3),
                })

            return {
                "query": query,
                "results": results,
                "total_results": len(results),
                "source": "tavily",
            }

        except Exception as e:
            logger.warning(
                "search_tool.tavily_error",
                error=str(e),
                message="Tavily call failed — returning empty results",
            )
            tool_calls_total.labels(tool_name="search", status="error").inc()
            return self._stub_result(query, error=str(e))

    def _stub_result(self, query: str, error: str | None = None) -> dict:
        """
        Minimal fallback when Tavily is unavailable.
        Returns an honest empty result rather than fake data.
        """
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "source": "stub",
            "error": error or "TAVILY_API_KEY not configured",
        }

    async def summarize_results(self, results: dict) -> str:
        """Format search results into a context string for the LLM."""
        if not results.get("results"):
            reason = results.get("error", "No results returned")
            return f"No search results found. Reason: {reason}"

        lines = [f"Search results for: '{results['query']}'", ""]
        for i, r in enumerate(results["results"], 1):
            title = r.get("title", "")
            if title:
                lines.append(f"[{i}] {title}")
            lines.append(f"    {r['snippet']}")
            lines.append(f"    Source: {r['source']} (relevance: {r['relevance_score']})")
            lines.append("")

        return "\n".join(lines)
