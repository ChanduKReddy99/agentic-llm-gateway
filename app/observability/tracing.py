"""
Distributed Tracing — OpenTelemetry → Tempo
============================================
Every agent step, LLM call, and tool invocation gets a trace span.
Traces flow: App → OpenTelemetry Collector → Tempo → Grafana
"""
import functools
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_tracer: trace.Tracer | None = None


def setup_tracing() -> None:
    """Initialize OpenTelemetry tracing with OTLP exporter → Tempo."""
    global _tracer
    try:
        resource = Resource(attributes={
            SERVICE_NAME: settings.otel_service_name,
            "deployment.environment": settings.app_env,
            "service.version": "1.0.0",
        })

        provider = TracerProvider(resource=resource)

        # Export spans to Tempo via OTLP gRPC
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(provider)

        _tracer = trace.get_tracer(__name__)
        logger.info(
            "tracing.initialized",
            endpoint=settings.otel_exporter_otlp_endpoint,
            service=settings.otel_service_name,
        )
    except Exception as e:
        logger.warning("tracing.init_failed", error=str(e), detail="Traces disabled")
        _tracer = trace.get_tracer(__name__)  # Falls back to no-op tracer


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        return trace.get_tracer(__name__)
    return _tracer


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """
    Context manager for creating trace spans.

    Usage:
        with trace_span("research_agent.search", {"query": query}) as span:
            result = await search(query)
            span.set_attribute("result_count", len(result))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@asynccontextmanager
async def async_trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Async version of trace_span for async functions."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def get_current_trace_id() -> str:
    """Get the current trace ID for correlation with logs."""
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return "no-trace"


def get_current_span_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        return format(ctx.span_id, "016x")
    return "no-span"
