"""
Agent Orchestrator
==================

Production pipeline — clean and correct:

  User Query
      │
      ▼
  [1. Input Guardrails]   ← block bad input before gateway sees it
      │
      ▼
  [2. LiteLLM Proxy]      ← gateway: cache, routing, fallbacks, rate limit
      │  cache HIT  ──────────────────────────────────► return instantly
      │  approved
      ▼
  [3. Research Agent]     ─► LiteLLM Proxy ─► LLM Provider
      │
      ▼
  [4. Synthesis Agent]    ─► LiteLLM Proxy ─► LLM Provider
      │
      ▼
  [5. Output Guardrails]  ← validate before returning to user
      │
      ▼
  Return to user

RAGAS evaluation is NOT in this pipeline.
It runs as a separate offline/background job against stored logs.
See: scripts/run_ragas_eval.py
     Only active when APP_ENV=development or via the /api/v1/eval endpoint.
"""
import time
import uuid
from typing import Any

import structlog

from app.agents.research_agent import ResearchAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.config.settings import get_settings
from app.gateway.cache import CacheClient
from app.gateway.litellm_client import LiteLLMGatewayClient
from app.guardrails.validator import InputGuardrails, OutputGuardrails
from app.observability.langfuse_tracker import LangfuseTracker
from app.observability.metrics import (
    active_pipelines,
    agent_pipeline_duration_seconds,
)
from app.observability.tracing import async_trace_span, get_current_trace_id
from app.tools.search_tool import SearchTool

logger = structlog.get_logger(__name__)
settings = get_settings()


class AgentOrchestrator:
    """
    Production pipeline coordinator.
    RAGAS evaluation is deliberately absent — it runs offline separately.
    """

    def __init__(self):
        self.cache = CacheClient()
        self.gateway = LiteLLMGatewayClient()
        self.langfuse = LangfuseTracker()
        self.search_tool = SearchTool(cache=self.cache)

        self.research_agent = ResearchAgent(
            gateway=self.gateway,
            search_tool=self.search_tool,
        )
        self.synthesis_agent = SynthesisAgent(gateway=self.gateway)

        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()

        logger.info("orchestrator.initialized")

    async def process(self, query: str, user_id: str | None = None) -> dict[str, Any]:
        request_id = str(uuid.uuid4())[:8]
        user_id = user_id or f"anon-{request_id}"
        start_time = time.time()

        active_pipelines.inc()

        langfuse_trace = self.langfuse.create_trace(
            name="agentic_pipeline",
            user_id=user_id,
            session_id=user_id,
            metadata={"request_id": request_id, "query": query[:200]},
            tags=["agentic", settings.app_env],
        )

        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            user_id=user_id,
        )

        async with async_trace_span(
            "orchestrator.pipeline",
            {"request_id": request_id, "query": query[:100]}
        ) as span:
            trace_id = get_current_trace_id()
            logger.info("pipeline.started", query=query[:100], trace_id=trace_id)

            try:
                # ── Phase 1: Input Guardrails ──────────────────────────────
                # Cheapest check first. Blocked here → gateway never sees it.
                logger.info("pipeline.phase", phase="1_input_guardrails")

                if settings.enable_input_guardrails:
                    input_result = await self.input_guardrails.validate(query)

                    if input_result.blocked:
                        duration = time.time() - start_time
                        agent_pipeline_duration_seconds.labels(
                            status="guardrail_blocked"
                        ).observe(duration)
                        logger.warning(
                            "pipeline.blocked",
                            phase="input_guardrails",
                            violations=[v["type"] for v in input_result.violations],
                        )
                        return {
                            "response": (
                                "⚠️ Your request was blocked by safety guardrails. "
                                "Please rephrase your question."
                            ),
                            "blocked": True,
                            "blocked_at": "input_guardrails",
                            "violations": input_result.violations,
                            "request_id": request_id,
                            "trace_id": trace_id,
                        }

                    safe_query = input_result.sanitized_text
                else:
                    safe_query = query

                # ── Phase 2: Gateway health check ─────────────────────────
                # Quick ping to LiteLLM proxy before spawning agents.
                # If proxy is down we log a warning but continue —
                # litellm_client.py has its own error handling per call.
                # NOTE: The actual gateway features (semantic cache, routing,
                # fallbacks, cost tracking) are NOT invoked here. They kick in
                # transparently inside each agent when self.gateway.chat_completion()
                # is called — i.e. the proxy sits between agents and LLM provider.
                logger.info("pipeline.phase", phase="2_gateway_health_check")

                gw_health = await self.gateway.health_check()
                span.set_attribute("gateway.status", gw_health.get("status", "unknown"))
                if gw_health.get("status") != "healthy":
                    logger.warning("pipeline.gateway_degraded", health=gw_health)

                # ── Phase 3: Research Agent ────────────────────────────────
                # Only reached after gateway approves.
                logger.info("pipeline.phase", phase="3_research_agent")

                research_output = await self.research_agent.run(
                    query=safe_query,
                    langfuse_trace=langfuse_trace,
                    request_id=request_id,
                )

                # ── Phase 4: Synthesis Agent ───────────────────────────────
                logger.info("pipeline.phase", phase="4_synthesis_agent")

                synthesis_output = await self.synthesis_agent.run(
                    query=safe_query,
                    research_output=research_output,
                    langfuse_trace=langfuse_trace,
                    request_id=request_id,
                )

                final_response = synthesis_output.get("final_response", "")
                sources = research_output.get("sources", [])

                # ── Phase 5: Output Guardrails ─────────────────────────────
                # Validate what agents produced before returning to user.
                logger.info("pipeline.phase", phase="5_output_guardrails")

                output_violations = []
                if settings.enable_output_guardrails:
                    output_result = await self.output_guardrails.validate(
                        text=final_response,
                        query=safe_query,
                    )
                    output_violations = output_result.violations
                    if output_violations:
                        logger.warning(
                            "pipeline.output_violations",
                            violations=[v["type"] for v in output_violations],
                        )

                # ── Dataset logging (Langfuse) ────────────────────────────
                # Log production Q&A pairs to Langfuse dataset.
                # Use this dataset to run RAGAS evals when prompt versions change.
                self.langfuse.log_to_dataset(
                    dataset_name="production_qa_pairs",
                    input_data={"query": safe_query, "contexts": [
                        r.get("snippet", "")
                        for sr in research_output.get("raw_search_results", [])
                        for r in sr.get("results", [])
                    ]},
                    expected_output=final_response,
                    metadata={"request_id": request_id, "env": settings.app_env},
                )

                # ── Done ──────────────────────────────────────────────────
                duration = time.time() - start_time
                total_tokens = (
                    sum(v.get("total_tokens", 0)
                        for v in research_output.get("token_usage", {}).values())
                    + sum(v.get("total_tokens", 0)
                          for v in synthesis_output.get("token_usage", {}).values())
                )
                cache_hits = (
                    research_output.get("cache_hits", 0)
                    + synthesis_output.get("cache_hits", 0)
                )
                # Total cost = sum of every LLM call across both agents.
                # Each individual call cost was already logged to Langfuse
                # and Prometheus by litellm_client.py. This is the rollup.
                total_cost_usd = (
                    research_output.get("cost_usd", 0.0)
                    + synthesis_output.get("cost_usd", 0.0)
                )

                agent_pipeline_duration_seconds.labels(status="success").observe(duration)

                logger.info(
                    "pipeline.completed",
                    duration_seconds=f"{duration:.3f}",
                    total_tokens=total_tokens,
                    total_cost_usd=f"${total_cost_usd:.6f}",
                    llm_cache_hits=cache_hits,
                    sources=len(sources),
                )

                return {
                    "response": final_response,
                    "sources": sources,
                    "blocked": False,
                    "pipeline_stats": {
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "duration_seconds": round(duration, 3),
                        "total_tokens": total_tokens,
                        "llm_cache_hits": cache_hits,
                        "models_used": {
                            "research": settings.research_agent_model,
                            "synthesis": settings.synthesis_agent_model,
                        },
                        "sub_queries": research_output.get("sub_queries", []),
                        "was_revised": synthesis_output.get("was_revised", False),
                        "output_violations": len(output_violations),
                        # Cost breakdown — how LiteLLM tracks this:
                        # proxy intercepts every agent LLM call, reads token
                        # counts from the response, multiplies by model price,
                        # logs to Langfuse. We roll it up here for the API response.
                        "total_cost_usd": round(total_cost_usd, 6),
                        "cost_breakdown": {
                            "research_agent": round(research_output.get("cost_usd", 0.0), 6),
                            "synthesis_agent": round(synthesis_output.get("cost_usd", 0.0), 6),
                        },
                        # RAGAS scores deliberately absent here.
                        # Run: scripts/run_ragas_eval.py
                    },
                    "request_id": request_id,
                    "trace_id": trace_id,
                }

            except Exception as e:
                duration = time.time() - start_time
                agent_pipeline_duration_seconds.labels(status="error").observe(duration)
                logger.error("pipeline.error", error=str(e), exc_info=True)
                return {
                    "response": f"An error occurred: {str(e)[:200]}",
                    "blocked": False,
                    "error": str(e),
                    "request_id": request_id,
                    "trace_id": trace_id,
                }

            finally:
                active_pipelines.dec()
                structlog.contextvars.clear_contextvars()
                self.langfuse.flush()
