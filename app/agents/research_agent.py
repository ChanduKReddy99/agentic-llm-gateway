"""
Research Agent — Agent 1
========================
Uses Langfuse prompt versioning: system prompts are fetched from
Langfuse registry so they can be changed without a code deploy.
Each agent run creates a Langfuse Span with individual Generations inside.
"""
import structlog

from app.config.settings import get_settings
from app.gateway.litellm_client import LiteLLMGatewayClient
from app.observability.metrics import agent_steps_total
from app.observability.tracing import async_trace_span
from app.tools.search_tool import SearchTool

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Hardcoded fallback prompts (used when Langfuse is unavailable) ────────────
# In production these live in Langfuse UI under "Prompts" and are versioned.
# Change them there without touching code.
_RESEARCH_SYSTEM_FALLBACK = """You are a specialized Research Agent. Your job is to:
1. Analyze the user's question to identify key information needs
2. Break it into 2-3 targeted search queries
3. Return a structured research brief

Output format:
## Research Brief
**Query Analysis**: [what the user is really asking]
**Key Findings**: [bullet points of most important facts]
**Context for Synthesis**: [paragraph of rich context]
**Sources**: [list the sources found]
**Confidence**: [High/Medium/Low with reason]
"""

_RESEARCH_QUERY_FALLBACK = "User Question: {question}\n\nIdentify the 2-3 most important sub-questions to research. List them briefly."

_RESEARCH_SYNTHESIS_FALLBACK = """User Question: {question}

Search Results:
{search_results}

Synthesize into a comprehensive Research Brief."""


class ResearchAgent:
    """
    Agent 1: Research & Information Gathering.

    Langfuse integration:
      - System prompts fetched by name+version from Langfuse registry
      - Each run() creates a Span inside the parent trace
      - Each LLM call tracked as a Generation linked to its prompt version
      - Costs visible per-generation in Langfuse UI
    """

    def __init__(self, gateway: LiteLLMGatewayClient, search_tool: SearchTool):
        self.gateway = gateway
        self.search_tool = search_tool
        self.agent_name = "research_agent"

    async def run(
        self,
        query: str,
        langfuse_trace=None,
        request_id: str = "unknown",
    ) -> dict:
        model = settings.research_agent_model

        async with async_trace_span(
            "agent.research", {"query": query[:100], "model": model}
        ) as span:
            logger.info("research_agent.started", query=query[:100], model=model, request_id=request_id)

            # ── Create a Langfuse Span for this agent ─────────────────────────
            # Span groups all generations from this agent together.
            # In Langfuse UI: Trace > research_agent span > [decompose, synthesis] generations
            lf_span = None
            if langfuse_trace:
                from app.observability.langfuse_tracker import LangfuseTracker
                lf = LangfuseTracker()
                lf_span = lf.create_span(
                    trace=langfuse_trace,
                    name="research_agent",
                    metadata={"model": model, "request_id": request_id},
                )

                # ── Fetch prompts from Langfuse registry ──────────────────────
                # Falls back to hardcoded strings if Langfuse is unreachable.
                # To version prompts: go to Langfuse UI → Prompts → create
                # "research_system_prompt" and "research_query_prompt"
                system_prompt_obj  = lf.get_prompt_object("research_system_prompt", label="production")
                system_prompt_text = lf.get_prompt("research_system_prompt", label="production", fallback=_RESEARCH_SYSTEM_FALLBACK)
                query_prompt_obj   = lf.get_prompt_object("research_query_prompt", label="production")
                query_prompt_text  = lf.get_prompt("research_query_prompt", label="production", fallback=_RESEARCH_QUERY_FALLBACK)
                synth_prompt_obj   = lf.get_prompt_object("research_synthesis_prompt", label="production")
                synth_prompt_text  = lf.get_prompt("research_synthesis_prompt", label="production", fallback=_RESEARCH_SYNTHESIS_FALLBACK)
            else:
                system_prompt_text = _RESEARCH_SYSTEM_FALLBACK
                query_prompt_text  = _RESEARCH_QUERY_FALLBACK
                synth_prompt_text  = _RESEARCH_SYNTHESIS_FALLBACK
                system_prompt_obj = query_prompt_obj = synth_prompt_obj = None

            all_search_results = []

            # ── Step 1: Decompose ─────────────────────────────────────────────
            agent_steps_total.labels(agent=self.agent_name, step_type="decomposition").inc()

            decomp_messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user",   "content": query_prompt_text.format(question=query)},
            ]

            decomp_response = await self.gateway.chat_completion(
                messages=decomp_messages,
                model=model,
                agent_name=self.agent_name,
                temperature=0.3,
                max_tokens=300,
                metadata={"step": "decomposition", "request_id": request_id},
            )

            # Track generation: links to prompt version + shows cost in Langfuse
            if lf_span:
                lf.track_generation(
                    trace=lf_span,
                    name="research_agent.decompose",
                    model=model,
                    messages=decomp_messages,
                    response=decomp_response,
                    agent_name=self.agent_name,
                    prompt_obj=query_prompt_obj,   # ← links this generation to prompt version
                )

            sub_queries = self._extract_queries(decomp_response.get("content", ""), query)

            # ── Step 2: Search ────────────────────────────────────────────────
            for sub_query in sub_queries[:3]:
                agent_steps_total.labels(agent=self.agent_name, step_type="search").inc()
                result = await self.search_tool.search(sub_query, max_results=3)
                all_search_results.append(result)

            # ── Step 3: Synthesise research brief ─────────────────────────────
            agent_steps_total.labels(agent=self.agent_name, step_type="synthesis").inc()

            combined = "\n\n---\n\n".join([
                await self.search_tool.summarize_results(r) for r in all_search_results
            ])

            synthesis_messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user",   "content": synth_prompt_text.format(
                    question=query, search_results=combined
                )},
            ]

            synthesis_response = await self.gateway.chat_completion(
                messages=synthesis_messages,
                model=model,
                agent_name=self.agent_name,
                temperature=0.2,
                max_tokens=800,
                metadata={"step": "synthesis", "request_id": request_id},
            )

            if lf_span:
                lf.track_generation(
                    trace=lf_span,
                    name="research_agent.synthesize",
                    model=model,
                    messages=synthesis_messages,
                    response=synthesis_response,
                    agent_name=self.agent_name,
                    prompt_obj=synth_prompt_obj,
                )

            all_sources = list({
                r.get("source", "")
                for sr in all_search_results
                for r in sr.get("results", [])
            })

            total_cost = (
                decomp_response.get("cost_usd", 0.0)
                + synthesis_response.get("cost_usd", 0.0)
            )

            span.set_attribute("sub_queries", len(sub_queries))
            span.set_attribute("cost_usd", total_cost)

            result = {
                "research_brief": synthesis_response.get("content", ""),
                "raw_search_results": all_search_results,
                "sources": all_sources,
                "sub_queries": sub_queries,
                "token_usage": {
                    "decomposition": decomp_response.get("usage", {}),
                    "synthesis":     synthesis_response.get("usage", {}),
                },
                "cache_hits": (
                    int(decomp_response.get("cache_hit", False))
                    + int(synthesis_response.get("cache_hit", False))
                ),
                "cost_usd": total_cost,
            }

            logger.info(
                "research_agent.completed",
                sub_queries=len(sub_queries),
                sources=len(all_sources),
                cache_hits=result["cache_hits"],
                cost_usd=f"${total_cost:.6f}",
            )
            return result

    def _extract_queries(self, text: str, original_query: str) -> list[str]:
        queries = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(("-", "*", "•"))):
                q = line.lstrip("0123456789.-*•) ").strip()
                if len(q) > 10:
                    queries.append(q)
        if not queries:
            queries = [original_query]
        elif original_query not in queries:
            queries.insert(0, original_query)
        return queries[:3]
