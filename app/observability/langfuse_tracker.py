"""
Langfuse Tracker — Full LLM Observability
==========================================

What Langfuse gives you that Prometheus/Grafana cannot:

  1. PROMPT VERSIONING
     Store prompts in Langfuse UI, fetch by name+version in code.
     Change a prompt without a code deploy. A/B test versions.
     Every generation links back to the exact prompt version used.

  2. PER-CALL COST BREAKDOWN
     Every LLM generation logs: model, prompt_tokens, completion_tokens,
     calculated cost_usd. Filter by agent, user, session, date.
     See exactly which agent/step is burning the most money.

  3. TRACE HIERARCHY
     Trace (one user request)
       └── Span: research_agent
             └── Generation: decompose    ← prompt v2, 310 tokens, $0.000047
             └── Generation: synthesis    ← prompt v1, 820 tokens, $0.000123
       └── Span: synthesis_agent
             └── Generation: draft        ← prompt v3, 1100 tokens, $0.000165
             └── Generation: critique     ← prompt v2,  580 tokens, $0.000087
     Total request cost = sum of all generations in trace.

  4. SCORES / EVALS ATTACHED TO TRACES
     RAGAS scores (faithfulness, relevancy) are attached to the trace
     so you can correlate: "low faithfulness → which prompt version?"

  5. DATASETS
     Log input/output pairs to a Langfuse dataset. Run evals against
     the dataset when you change a prompt version. CI quality gate.

Langfuse UI: http://localhost:3001
"""
from typing import Any

import structlog

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_langfuse = None


def _get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
            debug=False,
        )
        logger.info("langfuse.connected", host=settings.langfuse_host)
        return _langfuse
    except Exception as e:
        logger.warning("langfuse.unavailable", error=str(e))
        return None


class LangfuseTracker:
    """
    Full Langfuse integration: tracing, prompt versioning, cost, evals, datasets.
    All methods fail silently if Langfuse server is unreachable.
    """

    def __init__(self):
        self.lf = _get_langfuse()

    # ==== Prompt versioning ==========================================
    def get_prompt(
        self,
        name: str,
        version: int | None = None,
        label: str = "production",
        fallback: str = "",
    ) -> str:
        """
        Fetch a prompt from Langfuse prompt registry by name + version.

        Usage:
          prompt = langfuse.get_prompt("research_system_prompt", version=2)

        This lets you:
          - Change prompts from the Langfuse UI without a code deploy
          - A/B test prompt versions by toggling version= here
          - Every generation auto-links to the fetched prompt version

        Falls back to `fallback` string if Langfuse is unavailable.
        """
        if not self.lf:
            return fallback
        try:
            if version:
                prompt_obj = self.lf.get_prompt(name, version=version)
            else:
                prompt_obj = self.lf.get_prompt(name, label=label)
            text = prompt_obj.prompt   # raw template text
            logger.debug(
                "langfuse.prompt_fetched",
                name=name,
                label=label,
                version=getattr(prompt_obj, "version", "?"),
            )
            return text
        except Exception as e:
            logger.warning(
                "langfuse.prompt_fetch_failed",
                name=name,
                error=str(e),
                fallback="using hardcoded fallback",
            )
            return fallback

    def get_prompt_object(
        self,
        name: str,
        version: int | None = None,
        label: str = "production",
    ):
        """
        Return the raw Langfuse prompt object (for linking to generations).
        Agents always fetch label=production at runtime.
        CI/CD pushes new versions as label=staging first, then promotes to production.
        """
        if not self.lf:
            return None
        try:
            if version:
                return self.lf.get_prompt(name, version=version)
            return self.lf.get_prompt(name, label=label)
        except Exception:
            return None

    # ==== Trace lifecycle =====================================================

    def create_trace(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """
        Create a root Trace for one user request.

        In Langfuse UI this appears as a single row with:
          - total cost (sum of all generations inside)
          - total latency
          - user_id for per-user cost attribution
          - tags for filtering (e.g. "production", "dev")
        """
        if not self.lf:
            return _NoOpTrace()
        try:
            return self.lf.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or [],
            )
        except Exception as e:
            logger.warning("langfuse.trace_error", error=str(e))
            return _NoOpTrace()

    def create_span(self, trace, name: str, metadata: dict | None = None) -> Any:
        """
        Create a Span inside a Trace — one per agent.

        Spans group all generations for one agent together so you can see:
          research_agent total cost = decompose + synthesis generations
        """
        if not self.lf or isinstance(trace, _NoOpTrace):
            return _NoOpTrace()
        try:
            return trace.span(name=name, metadata=metadata or {})
        except Exception as e:
            logger.warning("langfuse.span_error", error=str(e))
            return _NoOpTrace()

    # ==== Generation tracking (LLM calls) =====================================

    def track_generation(
        self,
        trace,
        name: str,
        model: str,
        messages: list[dict],
        response: dict,
        agent_name: str = "unknown",
        prompt_obj=None,          # Langfuse prompt object for version linking
    ) -> None:
        """
        Record one LLM generation inside a trace.

        Langfuse auto-calculates cost from model + token counts using its
        built-in price table (same table LiteLLM proxy uses).

        What you see in Langfuse UI for this generation:
          - Prompt (full messages array, version linked if prompt_obj passed)
          - Completion (response text)
          - Model name
          - Prompt tokens / completion tokens / total tokens
          - Cost in USD (auto-calculated)
          - Latency
          - Cache hit flag
          - Agent name tag
        """
        if not self.lf or isinstance(trace, _NoOpTrace):
            return
        try:
            usage = response.get("usage", {})
            cost_usd = response.get("cost_usd", 0.0)

            gen_kwargs = dict(
                name=name,
                model=model,
                input=messages,
                output=response.get("content", ""),
                usage={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                    "unit": "TOKENS",
                },
                # Langfuse uses this to calculate cost if cost not provided
                # We also pass cost_usd explicitly for accuracy
                metadata={
                    "agent": agent_name,
                    "step": name.split(".")[-1],       # e.g. "decompose"
                    "cache_hit": response.get("cache_hit", False),
                    "cost_usd": cost_usd,
                    "latency_seconds": response.get("latency_seconds", 0),
                    "finish_reason": response.get("finish_reason", ""),
                },
            )

            # Link to prompt version if we fetched from Langfuse prompt registry
            if prompt_obj is not None:
                gen_kwargs["prompt"] = prompt_obj

            trace.generation(**gen_kwargs)

            logger.debug(
                "langfuse.generation_tracked",
                name=name,
                model=model,
                total_tokens=usage.get("total_tokens", 0),
                cost_usd=f"${cost_usd:.6f}",
                cache_hit=response.get("cache_hit", False),
            )
        except Exception as e:
            logger.warning("langfuse.generation_error", name=name, error=str(e))

    # ==== Scores / evals =======================================================

    def score_trace(
        self,
        trace,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """
        Attach a numeric score to a trace.

        Used for RAGAS metrics (offline eval script attaches scores here):
          langfuse.score_trace(trace, "ragas_faithfulness", 0.87)
          langfuse.score_trace(trace, "ragas_answer_relevancy", 0.92)

        In Langfuse UI you can then:
          - Filter traces by score range
          - Correlate low faithfulness with specific prompt versions
          - Plot score distributions over time
        """
        if not self.lf or isinstance(trace, _NoOpTrace):
            return
        try:
            trace.score(name=name, value=value, comment=comment)
        except Exception as e:
            logger.warning("langfuse.score_error", name=name, error=str(e))

    # ==== Dataset logging =======================================================

    def log_to_dataset(
        self,
        dataset_name: str,
        input_data: dict,
        expected_output: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Log a query+response pair to a Langfuse dataset.

        Use this to build a golden dataset of good Q&A pairs.
        Then run RAGAS eval against the dataset when you change prompt versions.

        Example:
          langfuse.log_to_dataset(
              dataset_name="production_qa_pairs",
              input_data={"query": query, "contexts": contexts},
              expected_output=final_response,
          )
        """
        if not self.lf:
            return
        try:
            # Get or create dataset
            try:
                self.lf.get_dataset(dataset_name)
            except Exception:
                self.lf.create_dataset(
                    name=dataset_name,
                    description="Auto-logged production Q&A pairs",
                )

            self.lf.create_dataset_item(
                dataset_name=dataset_name,
                input=input_data,
                expected_output=expected_output,
                metadata=metadata or {},
            )
            logger.debug("langfuse.dataset_logged", dataset=dataset_name)
        except Exception as e:
            logger.warning("langfuse.dataset_error", error=str(e))

    # ====  Flush ============================================================

    def flush(self) -> None:
        """Flush all pending events to Langfuse. Call at request end."""
        if self.lf:
            try:
                self.lf.flush()
            except Exception:
                pass


class _NoOpTrace:
    """Null object — code never breaks when Langfuse is unreachable."""
    def generation(self, **kwargs): pass
    def score(self, **kwargs): pass
    def span(self, **kwargs): return self
    def __enter__(self): return self
    def __exit__(self, *args): pass
