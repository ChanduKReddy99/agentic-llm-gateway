"""
Tests for Langfuse Tracker — prompt versioning, traces, cost tracking, datasets.
All tests use the NoOp fallback path (Langfuse server not required).
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.observability.langfuse_tracker import LangfuseTracker, _NoOpTrace


@pytest.fixture
def tracker_no_server():
    """Tracker with no Langfuse server — uses NoOp fallback silently."""
    with patch("app.observability.langfuse_tracker._get_langfuse", return_value=None):
        t = LangfuseTracker()
        t.lf = None   # force NoOp path
        return t


class TestLangfuseTrackerNoOp:
    """When Langfuse is unreachable every method must be a safe no-op."""

    def test_create_trace_returns_noop(self, tracker_no_server):
        trace = tracker_no_server.create_trace(name="test", user_id="u1")
        assert isinstance(trace, _NoOpTrace)

    def test_create_span_returns_noop(self, tracker_no_server):
        trace = _NoOpTrace()
        span = tracker_no_server.create_span(trace=trace, name="research_agent")
        assert isinstance(span, _NoOpTrace)

    def test_track_generation_does_not_raise(self, tracker_no_server):
        """Must never raise even with real-looking data."""
        trace = _NoOpTrace()
        tracker_no_server.track_generation(
            trace=trace,
            name="research_agent.decompose",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            response={
                "content": "LiteLLM is a gateway.",
                "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                "cost_usd": 0.0000075,
                "cache_hit": False,
                "latency_seconds": 0.4,
                "finish_reason": "stop",
            },
            agent_name="research_agent",
        )

    def test_score_trace_does_not_raise(self, tracker_no_server):
        trace = _NoOpTrace()
        tracker_no_server.score_trace(trace=trace, name="ragas_faithfulness", value=0.87)

    def test_get_prompt_returns_fallback(self, tracker_no_server):
        """When Langfuse unavailable, get_prompt must return the fallback string."""
        fallback = "You are a research agent."
        result = tracker_no_server.get_prompt("research_system_prompt", fallback=fallback)
        assert result == fallback

    def test_get_prompt_object_returns_none(self, tracker_no_server):
        result = tracker_no_server.get_prompt_object("any_prompt")
        assert result is None

    def test_log_to_dataset_does_not_raise(self, tracker_no_server):
        tracker_no_server.log_to_dataset(
            dataset_name="production_qa_pairs",
            input_data={"query": "What is LiteLLM?", "contexts": ["LiteLLM is a gateway."]},
            expected_output="LiteLLM provides a unified LLM API.",
        )

    def test_flush_does_not_raise(self, tracker_no_server):
        tracker_no_server.flush()


class TestNoOpTrace:
    """NoOp trace object must absorb all method calls safely."""

    def test_generation_absorbs_call(self):
        t = _NoOpTrace()
        t.generation(name="test", model="gpt-4o-mini", input=[], output="hi")

    def test_score_absorbs_call(self):
        t = _NoOpTrace()
        t.score(name="faithfulness", value=0.9)

    def test_span_returns_self(self):
        t = _NoOpTrace()
        result = t.span(name="agent_span")
        assert isinstance(result, _NoOpTrace)

    def test_context_manager_works(self):
        t = _NoOpTrace()
        with t as inner:
            assert inner is t


class TestLangfuseTrackerWithMockServer:
    """Tests with a mocked Langfuse client to verify real call signatures."""

    @pytest.fixture
    def mock_lf(self):
        return MagicMock()

    @pytest.fixture
    def tracker_with_mock(self, mock_lf):
        t = LangfuseTracker()
        t.lf = mock_lf
        return t

    def test_create_trace_calls_lf_trace(self, tracker_with_mock, mock_lf):
        tracker_with_mock.create_trace(
            name="agentic_pipeline",
            user_id="user-1",
            session_id="sess-1",
            metadata={"request_id": "abc"},
            tags=["production"],
        )
        mock_lf.trace.assert_called_once()
        call_kwargs = mock_lf.trace.call_args[1]
        assert call_kwargs["name"] == "agentic_pipeline"
        assert call_kwargs["user_id"] == "user-1"
        assert call_kwargs["tags"] == ["production"]

    def test_score_trace_calls_score_on_trace(self, tracker_with_mock):
        mock_trace = MagicMock()
        tracker_with_mock.score_trace(mock_trace, "ragas_faithfulness", 0.87, "Good")
        mock_trace.score.assert_called_once_with(
            name="ragas_faithfulness", value=0.87, comment="Good"
        )

    def test_track_generation_calls_generation_on_trace(self, tracker_with_mock):
        mock_trace = MagicMock()
        tracker_with_mock.track_generation(
            trace=mock_trace,
            name="research_agent.decompose",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            response={
                "content": "result",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                "cost_usd": 0.000022,
                "cache_hit": False,
                "latency_seconds": 0.3,
                "finish_reason": "stop",
            },
            agent_name="research_agent",
        )
        mock_trace.generation.assert_called_once()
        gen_kwargs = mock_trace.generation.call_args[1]
        assert gen_kwargs["name"] == "research_agent.decompose"
        assert gen_kwargs["model"] == "gpt-4o-mini"
        assert gen_kwargs["usage"]["input"] == 100
        assert gen_kwargs["usage"]["output"] == 50

    def test_get_prompt_calls_lf_get_prompt(self, tracker_with_mock, mock_lf):
        mock_prompt = MagicMock()
        mock_prompt.prompt = "You are a research agent v2."
        mock_lf.get_prompt.return_value = mock_prompt

        result = tracker_with_mock.get_prompt(
            "research_system_prompt",
            fallback="fallback text"
        )
        mock_lf.get_prompt.assert_called_once_with("research_system_prompt", label="production")
        assert result == "You are a research agent v2."

    def test_get_prompt_falls_back_on_exception(self, tracker_with_mock, mock_lf):
        mock_lf.get_prompt.side_effect = Exception("prompt not found")
        result = tracker_with_mock.get_prompt(
            "nonexistent_prompt",
            fallback="my fallback"
        )
        assert result == "my fallback"
