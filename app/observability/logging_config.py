"""
Structured Logging — structlog → Loki (direct push)
=====================================================

No Promtail needed. The app pushes logs directly to Loki
using python-logging-loki as a logging handler.

Why no Promtail here:
  - Promtail is a log shipper for collecting logs from FILES or
    container stdout on a node. It's needed in Kubernetes where
    pods write to disk and something must collect + ship them.
  - Here we run a Python app locally. We can push logs DIRECTLY
    to Loki's HTTP endpoint from inside the app.
  - Simpler: one fewer container, no Docker socket needed,
    logs go straight to Loki with full structured fields.

Flow:
  structlog event
      │
      ├──► ConsoleRenderer (dev)    → pretty terminal output
      └──► LokiHandler              → POST /loki/api/v1/push
                                       (direct HTTP to Loki)

Every log line carries:
  - trace_id / span_id  → correlates with Tempo traces in Grafana
  - agent               → which agent produced this log
  - request_id          → correlates all logs for one user request
  - level               → info / warning / error
  - service             → agentic-ai-app
"""
import logging
import sys
from typing import Any

import structlog

from app.config.settings import get_settings

settings = get_settings()


def setup_logging() -> None:
    """
    Configure structlog with direct Loki push (no Promtail).
    Falls back gracefully if Loki is unreachable.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_trace_context,
        _add_service_context,
    ]

    if settings.app_env == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging for third-party libraries
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.log_level),
    )

    # Silence noisy libraries
    for noisy in ["httpx", "openai", "litellm", "urllib3", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Wire direct Loki push handler
    _setup_loki_handler()


def _setup_loki_handler() -> None:
    """
    Attach a Loki HTTP handler to the root logger.
    Pushes log records directly to Loki — no Promtail needed.
    Falls back silently if Loki is unreachable or package missing.
    """
    try:
        import logging_loki  # type: ignore

        loki_url = f"{settings.loki_url}/loki/api/v1/push"

        handler = logging_loki.LokiHandler(
            url=loki_url,
            tags={
                "service":     settings.otel_service_name,
                "environment": settings.app_env,
            },
            auth=None,          # no auth for local Loki
            version="1",
        )
        handler.setLevel(logging.getLevelName(settings.log_level))

        # Attach to root logger so ALL log output goes to Loki
        logging.getLogger().addHandler(handler)

        structlog.get_logger(__name__).info(
            "loki.handler_attached",
            url=loki_url,
            message="Logs pushing directly to Loki (no Promtail)",
        )

    except ImportError:
        structlog.get_logger(__name__).warning(
            "loki.handler_skipped",
            reason="logging-loki package not installed",
            message="Install with: uv add python-logging-loki",
        )
    except Exception as e:
        structlog.get_logger(__name__).warning(
            "loki.handler_failed",
            error=str(e),
            message="Loki unreachable — logs go to stdout only",
        )


def _add_trace_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    """
    Inject OpenTelemetry trace_id + span_id into every log record.
    This is how Grafana links a log line to its trace in Tempo:
      Log line has trace_id → click → opens Tempo trace waterfall.
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx  = span.get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"]  = format(ctx.span_id,  "016x")
    except Exception:
        pass
    return event_dict


def _add_service_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Add service name to every log — used as a Loki label for filtering."""
    event_dict["service"] = settings.otel_service_name
    return event_dict
