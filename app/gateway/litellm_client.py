"""
LiteLLM Gateway Client
======================

HOW LITELLM TRACKS COST — step by step:
-----------------------------------------

Every agent call (decompose, synthesis, draft, critique) hits this client.
Each call flows:

  agent.chat_completion()
       │
       ▼
  LiteLLM Proxy (HTTP POST /chat/completions)
       │
       │  1. Proxy receives the request
       │  2. Checks Redis semantic cache → HIT: return cached, cost = $0
       │  3. MISS: forwards to LLM provider (OpenAI / Anthropic / Groq)
       │  4. Gets response back with usage: { prompt_tokens, completion_tokens }
       │  5. Looks up cost_per_token from its built-in model price table:
       │       e.g. gpt-4o-mini: input=$0.00015/1K, output=$0.0006/1K
       │  6. Calculates:
       │       cost = (prompt_tokens/1000 * input_price)
       │            + (completion_tokens/1000 * output_price)
       │  7. Logs to Langfuse callback: { model, tokens, cost, agent_name, step }
       │  8. Returns response + usage to this client
       │
       ▼
  This client reads response.usage (tokens)
  We also compute cost locally here for immediate Prometheus metrics.
  LiteLLM proxy does the same independently for Langfuse.

So cost is tracked TWICE, giving you two views:
  • Prometheus/Grafana  — real-time, per-agent, per-step aggregates
  • Langfuse            — per-individual-call drill-down with full prompt/response

The agent code itself does NOTHING special for cost tracking.
It just calls chat_completion() — the proxy and this client handle everything.

Model price table used by LiteLLM proxy (built-in, auto-updated):
  https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""
import time
from typing import Any

import httpx
import structlog
from openai import AsyncOpenAI

from app.config.settings import get_settings
from app.observability.metrics import (
    gateway_cost_usd_total,
    gateway_cache_hits_total,
    gateway_requests_total,
    gateway_tokens_total,
    llm_latency_seconds,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Cost per 1K tokens (USD) — mirrors LiteLLM's internal price table.
# Used here for local Prometheus cost metrics.
# LiteLLM proxy uses its own built-in table for Langfuse logging.
# ---------------------------------------------------------------------------
MODEL_COSTS_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini":                  {"input": 0.00015,  "output": 0.0006},
    "gpt-4o":                       {"input": 0.005,    "output": 0.015},
    "claude-3-haiku-20240307":      {"input": 0.00025,  "output": 0.00125},
    "claude-3-5-sonnet-20241022":   {"input": 0.003,    "output": 0.015},
    "groq/llama3-8b-8192":          {"input": 0.00005,  "output": 0.00008},
    "groq/llama3-70b-8192":         {"input": 0.00059,  "output": 0.00079},
}


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate USD cost for a single LLM call.

    This is the same formula LiteLLM proxy uses internally:
      cost = (prompt_tokens / 1000 * input_price_per_1k)
           + (completion_tokens / 1000 * output_price_per_1k)

    The proxy applies this AFTER every non-cached call and logs it to Langfuse.
    We apply it here too for immediate Prometheus visibility.
    """
    prices = MODEL_COSTS_PER_1K.get(model, {"input": 0.001, "output": 0.002})
    return (prompt_tokens / 1000 * prices["input"]) + (completion_tokens / 1000 * prices["output"])


class LiteLLMGatewayClient:
    """
    Routes ALL LLM calls through the LiteLLM proxy.

    From the agent's perspective: just call chat_completion().
    The proxy transparently handles caching, routing, fallbacks,
    rate limiting, and cost logging — agents know nothing about this.

    Cost tracking happens at two levels:
      1. LiteLLM proxy → Langfuse  (per-call detail, automatic)
      2. This client   → Prometheus (per-agent aggregates, real-time)
    """

    def __init__(self):
        self.settings = get_settings()
        # Standard OpenAI client pointed at LiteLLM proxy.
        # LiteLLM is OpenAI-API compatible — no special SDK needed.
        self.client = AsyncOpenAI(
            api_key=self.settings.litellm_master_key,
            base_url=self.settings.litellm_proxy_url,
        )

    async def chat_completion(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        agent_name: str = "unknown",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """
        Single entry point for ALL LLM calls from ALL agents.

        Flow inside LiteLLM proxy for each call:
          1. Cache check (Redis semantic similarity)
             → HIT:  return cached response, cost = $0, agents_called++
             → MISS: forward to LLM provider
          2. Get LLM response with token usage
          3. Calculate cost from token counts × model price table
          4. Log { model, tokens, cost, agent_name, step } to Langfuse
          5. Return response to this client

        This client then:
          - Emits token counts to Prometheus (per agent, per model)
          - Calculates and emits cost_usd to Prometheus
          - Logs a structured line so Loki shows per-call cost
        """
        start_time = time.time()
        metadata = metadata or {}

        gateway_requests_total.labels(agent=agent_name, model=model).inc()

        try:
            logger.info(
                "gateway.request",
                agent=agent_name,
                model=model,
                step=metadata.get("step", "unknown"),
            )

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "x-agent-name": agent_name,
                    "x-request-id": metadata.get("request_id", "unknown"),
                },
                # These metadata fields flow into LiteLLM proxy logs and
                # Langfuse — this is how per-agent cost attribution works.
                extra_body={
                    "metadata": {
                        "agent_name": agent_name,         # ← Langfuse tags by this
                        "step": metadata.get("step"),     # ← e.g. "decomposition"
                        "request_id": metadata.get("request_id"),
                        "environment": self.settings.app_env,
                    }
                },
            )

            latency = time.time() - start_time
            usage = response.usage

            prompt_tokens     = usage.prompt_tokens     if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens      = usage.total_tokens      if usage else 0

            # Cost calculation — same formula as LiteLLM proxy uses internally.
            # The proxy logs this to Langfuse automatically.
            # We also push it to Prometheus for Grafana dashboards.
            cost_usd = _calculate_cost(model, prompt_tokens, completion_tokens)

            # ── Prometheus metrics ────────────────────────────────────────
            llm_latency_seconds.labels(agent=agent_name, model=model).observe(latency)
            gateway_tokens_total.labels(agent=agent_name, model=model, type="prompt").inc(prompt_tokens)
            gateway_tokens_total.labels(agent=agent_name, model=model, type="completion").inc(completion_tokens)

            gateway_cost_usd_total.labels(agent=agent_name, model=model).inc(cost_usd)

            # Cache hit detection — LiteLLM proxy sets this header on cached responses
            raw_headers = getattr(response, "_headers", {})
            cache_hit = str(raw_headers.get("x-litellm-cache-hit", "false")).lower() == "true"
            if cache_hit:
                gateway_cache_hits_total.labels(agent=agent_name).inc()

            logger.info(
                "gateway.response",
                agent=agent_name,
                step=metadata.get("step", "unknown"),
                model=response.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=f"${cost_usd:.6f}",   # visible in Loki logs per call
                cache_hit=cache_hit,
                latency_seconds=f"{latency:.3f}",
                # When cache_hit=True: cost_usd=$0.000000 — the saving is visible
            )

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "cost_usd": cost_usd,           # returned to agent, rolled up in pipeline_stats
                "cache_hit": cache_hit,
                "latency_seconds": latency,
                "finish_reason": response.choices[0].finish_reason,
            }

        except Exception as e:
            latency = time.time() - start_time
            llm_latency_seconds.labels(agent=agent_name, model=model).observe(latency)
            logger.error("gateway.error", agent=agent_name, model=model, error=str(e))
            return {
                "content": f"[Gateway Error] {str(e)[:200]}",
                "model": model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost_usd": 0.0,
                "cache_hit": False,
                "latency_seconds": latency,
                "finish_reason": "error",
                "error": str(e),
            }

    async def health_check(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.settings.litellm_proxy_url}/health")
                return {"status": "healthy", "proxy_status": resp.status_code}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_proxy_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.settings.litellm_proxy_url}/models",
                    headers={"Authorization": f"Bearer {self.settings.litellm_master_key}"},
                )
                return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return ["gpt-4o-mini", "claude-3-haiku-20240307", "groq/llama3-8b-8192"]

    async def get_spend_stats(self) -> dict:
        """
        Fetch per-agent spend breakdown from LiteLLM proxy.

        The proxy tracks this because we pass agent_name in metadata on
        every call. Langfuse also shows this as a cost breakdown by tag.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.settings.litellm_proxy_url}/spend/logs",
                    headers={"Authorization": f"Bearer {self.settings.litellm_master_key}"},
                )
                return resp.json()
        except Exception as e:
            return {"error": str(e), "message": "LiteLLM proxy not reachable"}
