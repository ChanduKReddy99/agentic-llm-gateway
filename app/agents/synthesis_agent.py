"""
Synthesis Agent — Agent 2
=========================
Langfuse prompt versioning + spans + generation cost tracking.
"""
import structlog

from app.config.settings import get_settings
from app.gateway.litellm_client import LiteLLMGatewayClient
from app.observability.metrics import agent_steps_total
from app.observability.tracing import async_trace_span

logger = structlog.get_logger(__name__)
settings = get_settings()

_SYNTHESIS_SYSTEM_FALLBACK = """You are a Synthesis Agent. Produce clear, accurate,
well-structured responses from research findings.

Guidelines:
- Directly answer the question first
- Support claims with evidence from the research
- Use markdown for readability
- Acknowledge gaps or uncertainties
- Include a Sources section
"""

_SYNTHESIS_DRAFT_FALLBACK = """Original Question: {question}

Research Brief:
{research_brief}

Additional context: {additional_context}

Synthesize into a comprehensive, well-structured response."""

_SYNTHESIS_CRITIQUE_FALLBACK = """Review this response for quality:

Question: {question}

Response:
{response}

Check: (1) directly answers question? (2) claims supported by research?
(3) clear and well-structured? (4) any errors?

If good, reply: "APPROVED: [reason]"
If revision needed, provide the improved response directly."""


class SynthesisAgent:
    """
    Agent 2: Response Synthesis + Self-Critique.

    Langfuse integration:
      - Prompts fetched by name+version from Langfuse registry
      - Span groups draft + critique generations together
      - Each generation linked to its prompt version for A/B testing
    """

    def __init__(self, gateway: LiteLLMGatewayClient):
        self.gateway = gateway
        self.agent_name = "synthesis_agent"

    async def run(
        self,
        query: str,
        research_output: dict,
        langfuse_trace=None,
        request_id: str = "unknown",
    ) -> dict:
        model = settings.synthesis_agent_model

        async with async_trace_span(
            "agent.synthesis", {"query": query[:100], "model": model}
        ) as span:
            logger.info("synthesis_agent.started", query=query[:100], model=model, request_id=request_id)

            # ── Langfuse span + prompt fetch ──────────────────────────────────
            lf_span = None
            if langfuse_trace:
                from app.observability.langfuse_tracker import LangfuseTracker
                lf = LangfuseTracker()
                lf_span = lf.create_span(
                    trace=langfuse_trace,
                    name="synthesis_agent",
                    metadata={"model": model, "request_id": request_id},
                )

                system_prompt_obj   = lf.get_prompt_object("synthesis_system_prompt", label="production")
                system_prompt_text  = lf.get_prompt("synthesis_system_prompt", label="production", fallback=_SYNTHESIS_SYSTEM_FALLBACK)
                draft_prompt_obj    = lf.get_prompt_object("synthesis_draft_prompt", label="production")
                draft_prompt_text   = lf.get_prompt("synthesis_draft_prompt", label="production", fallback=_SYNTHESIS_DRAFT_FALLBACK)
                critique_prompt_obj = lf.get_prompt_object("synthesis_critique_prompt", label="production")
                critique_prompt_text = lf.get_prompt("synthesis_critique_prompt", label="production", fallback=_SYNTHESIS_CRITIQUE_FALLBACK)
            else:
                system_prompt_text   = _SYNTHESIS_SYSTEM_FALLBACK
                draft_prompt_text    = _SYNTHESIS_DRAFT_FALLBACK
                critique_prompt_text = _SYNTHESIS_CRITIQUE_FALLBACK
                system_prompt_obj = draft_prompt_obj = critique_prompt_obj = None

            research_brief = research_output.get("research_brief", "No research available.")
            sources        = research_output.get("sources", [])
            additional_ctx = f"Sources: {', '.join(sources[:5])}" if sources else "No external sources."

            # ── Step 1: Draft ─────────────────────────────────────────────────
            agent_steps_total.labels(agent=self.agent_name, step_type="draft").inc()

            draft_messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user",   "content": draft_prompt_text.format(
                    question=query,
                    research_brief=research_brief,
                    additional_context=additional_ctx,
                )},
            ]

            draft_response = await self.gateway.chat_completion(
                messages=draft_messages,
                model=model,
                agent_name=self.agent_name,
                temperature=0.4,
                max_tokens=1200,
                metadata={"step": "draft", "request_id": request_id},
            )

            if lf_span:
                lf.track_generation(
                    trace=lf_span,
                    name="synthesis_agent.draft",
                    model=model,
                    messages=draft_messages,
                    response=draft_response,
                    agent_name=self.agent_name,
                    prompt_obj=draft_prompt_obj,
                )

            draft_content = draft_response.get("content", "")

            # ── Step 2: Self-critique ─────────────────────────────────────────
            agent_steps_total.labels(agent=self.agent_name, step_type="critique").inc()

            critique_messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user",   "content": critique_prompt_text.format(
                    question=query, response=draft_content,
                )},
            ]

            critique_response = await self.gateway.chat_completion(
                messages=critique_messages,
                model=model,
                agent_name=self.agent_name,
                temperature=0.2,
                max_tokens=600,
                metadata={"step": "critique", "request_id": request_id},
            )

            if lf_span:
                lf.track_generation(
                    trace=lf_span,
                    name="synthesis_agent.critique",
                    model=model,
                    messages=critique_messages,
                    response=critique_response,
                    agent_name=self.agent_name,
                    prompt_obj=critique_prompt_obj,
                )

            critique_content = critique_response.get("content", "")

            was_revised    = False
            final_content  = draft_content
            if (
                critique_content
                and "APPROVED" not in critique_content.upper()[:50]
                and len(critique_content) > len(draft_content) * 0.5
            ):
                final_content = critique_content
                was_revised   = True
                logger.info("synthesis_agent.revised")

            total_cost = (
                draft_response.get("cost_usd", 0.0)
                + critique_response.get("cost_usd", 0.0)
            )

            span.set_attribute("was_revised", was_revised)
            span.set_attribute("cost_usd", total_cost)

            result = {
                "final_response": final_content,
                "draft_response": draft_content,
                "critique":       critique_content,
                "was_revised":    was_revised,
                "token_usage": {
                    "draft":    draft_response.get("usage", {}),
                    "critique": critique_response.get("usage", {}),
                },
                "cache_hits": (
                    int(draft_response.get("cache_hit", False))
                    + int(critique_response.get("cache_hit", False))
                ),
                "cost_usd": total_cost,
            }

            logger.info(
                "synthesis_agent.completed",
                was_revised=was_revised,
                cache_hits=result["cache_hits"],
                cost_usd=f"${total_cost:.6f}",
            )
            return result
