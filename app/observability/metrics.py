"""
Prometheus Metrics
==================
All application metrics defined here. Scraped by Prometheus,
visualized in Grafana dashboards.

Metric naming convention: {app}_{subsystem}_{metric}_{unit}
"""
from prometheus_client import Counter, Gauge, Histogram, Summary

# ===== LLM Gateway Metrics ===================================================

gateway_requests_total = Counter(
    "agentic_gateway_requests_total",
    "Total LLM requests through the gateway",
    labelnames=["agent", "model"],
)

gateway_cache_hits_total = Counter(
    "agentic_gateway_cache_hits_total",
    "Total semantic cache hits in LiteLLM proxy",
    labelnames=["agent"],
)

gateway_tokens_total = Counter(
    "agentic_gateway_tokens_total",
    "Total tokens consumed through the gateway",
    labelnames=["agent", "model", "type"],  # type: prompt | completion
)

llm_latency_seconds = Histogram(
    "agentic_llm_latency_seconds",
    "LLM response latency through gateway",
    labelnames=["agent", "model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

gateway_cost_usd_total = Counter(
    "agentic_gateway_cost_usd_total",
    "Total estimated USD cost of LLM calls through the gateway",
    labelnames=["agent", "model"],
)

# =====  Agent Pipeline Metrics ===============================================

agent_pipeline_duration_seconds = Histogram(
    "agentic_pipeline_duration_seconds",
    "End-to-end agentic pipeline duration",
    labelnames=["status"],  # success | error | guardrail_blocked
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

agent_steps_total = Counter(
    "agentic_agent_steps_total",
    "Total steps executed by each agent",
    labelnames=["agent", "step_type"],
)

agent_errors_total = Counter(
    "agentic_agent_errors_total",
    "Total errors in agent execution",
    labelnames=["agent", "error_type"],
)

active_pipelines = Gauge(
    "agentic_active_pipelines",
    "Currently running agent pipelines",
)

# ===== Guardrails Metrics ===================================================

guardrails_violations_total = Counter(
    "agentic_guardrails_violations_total",
    "Total guardrail violations detected",
    labelnames=["stage", "violation_type"],  # stage: input | output
)

guardrails_latency_seconds = Summary(
    "agentic_guardrails_latency_seconds",
    "Guardrails validation latency",
    labelnames=["stage"],
)

# ===== RAGAS Evaluation Metrics =============================================

ragas_faithfulness = Histogram(
    "agentic_ragas_faithfulness",
    "RAGAS faithfulness score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ragas_answer_relevancy = Histogram(
    "agentic_ragas_answer_relevancy",
    "RAGAS answer relevancy score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ragas_evaluations_total = Counter(
    "agentic_ragas_evaluations_total",
    "Total RAGAS evaluations run",
)

# ===== Tool Usage Metrics ===================================================

tool_calls_total = Counter(
    "agentic_tool_calls_total",
    "Total tool invocations by the research agent",
    labelnames=["tool_name", "status"],
)

tool_latency_seconds = Histogram(
    "agentic_tool_latency_seconds",
    "Tool execution latency",
    labelnames=["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
)
